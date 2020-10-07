export HostCall, hostcall!

const DEFAULT_HOSTCALL_LATENCY = 0.01

## Sentinels in order of execution

"Signal is ready for accessing by host or device."
const READY_SENTINEL = 0

"Device has locked the signal."
const DEVICE_LOCK_SENTINEL = 1

"Device-sourced message is available."
const DEVICE_MSG_SENTINEL = 2

"Host has locked the signal."
const HOST_LOCK_SENTINEL = 3

"Host-sourced message is available."
const HOST_MSG_SENTINEL = 4

## Error sentinels

"Fatal error on device wavefront accessing the signal."
const DEVICE_ERR_SENTINEL = 5

"Fatal error on host thread accessing the signal."
const HOST_ERR_SENTINEL = 6

"""
    HostCall{S,RT,AT}

GPU-compatible struct for making hostcalls.
"""
struct HostCall{S,RT,AT}
    signal::S
    buf_ptr::DevicePtr{UInt8,AS.Global}
    buf_len::UInt
end
function HostCall(RT::Type, AT::Type{<:Tuple}, signal::S;
                  agent=get_default_agent(), buf_size=nothing) where S
    @assert S == UInt64
    if buf_size === nothing
        buf_size = 0
        for T in AT.parameters
            @assert isbitstype(T) "Hostcall arguments must be bits-type"
            buf_size += sizeof(T)
        end
    end
    buf_size = max(sizeof(UInt64), buf_size) # make room for return buffer pointer
    buf = Mem.alloc(agent, buf_size)
    buf_ptr = DevicePtr{UInt8,AS.Global}(Base.unsafe_convert(Ptr{UInt8}, buf))
    HSA.signal_store_release(HSA.Signal(signal), READY_SENTINEL)
    memfence!(Val(:seq_cst))
    HostCall{S,RT,AT}(signal, buf_ptr, buf_size)
end

## device signal functions
# TODO: device_signal_load, device_signal_add!, etc.
@inline @generated function device_signal_store!(signal::UInt64, value::Int64)
    JuliaContext() do ctx
        T_nothing = convert(LLVMType, Nothing, ctx)
        T_i32 = LLVM.Int32Type(ctx)
        T_i64 = LLVM.Int64Type(ctx)

        # create a function
        llvm_f, _ = create_function(T_nothing, [T_i64, T_i64])
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            T_signal_store = LLVM.FunctionType(T_nothing, [T_i64, T_i64, T_i32])
            signal_store = LLVM.Function(mod, "__ockl_hsa_signal_store", T_signal_store)
            call!(builder, signal_store, [parameters(llvm_f)[1],
                                          parameters(llvm_f)[2],
                                          # __ATOMIC_RELEASE == 3
                                          ConstantInt(Int32(3), ctx)])

            ret!(builder)
        end

        call_function(llvm_f, Nothing, Tuple{UInt64,Int64}, :((signal,value)))
    end
end
@inline @generated function device_signal_wait(signal::UInt64, value::Int64)
    JuliaContext() do ctx
        T_nothing = convert(LLVMType, Nothing, ctx)
        T_i32 = LLVM.Int32Type(ctx)
        T_i64 = LLVM.Int64Type(ctx)

        # create a function
        llvm_f, _ = create_function(T_nothing, [T_i64, T_i64])
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            signal_match = BasicBlock(llvm_f, "signal_match", ctx)
            signal_miss = BasicBlock(llvm_f, "signal_miss", ctx)
            signal_miss_1 = BasicBlock(llvm_f, "signal_miss_1", ctx)
            signal_miss_2 = BasicBlock(llvm_f, "signal_miss_2", ctx)
            signal_fail = BasicBlock(llvm_f, "signal_fail", ctx)

            position!(builder, entry)
            br!(builder, signal_miss)

            position!(builder, signal_miss)
            T_sleep = LLVM.FunctionType(T_nothing, [T_i32])
            sleep_f = LLVM.Function(mod, "llvm.amdgcn.s.sleep", T_sleep)
            call!(builder, sleep_f, [ConstantInt(Int32(1), ctx)])
            T_signal_load = LLVM.FunctionType(T_i64, [T_i64, T_i32])
            signal_load = LLVM.Function(mod, "__ockl_hsa_signal_load", T_signal_load)
            loaded_value = call!(builder, signal_load, [parameters(llvm_f)[1],
                                                        # __ATOMIC_ACQUIRE == 2
                                                        ConstantInt(Int32(2), ctx)])
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, parameters(llvm_f)[2])
            br!(builder, cond, signal_match, signal_miss_1)

            position!(builder, signal_miss_1)
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, ConstantInt(Int64(DEVICE_ERR_SENTINEL), ctx))
            br!(builder, cond, signal_fail, signal_miss_2)

            position!(builder, signal_miss_2)
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, ConstantInt(Int64(HOST_ERR_SENTINEL), ctx))
            br!(builder, cond, signal_fail, signal_miss)

            position!(builder, signal_fail)
            call!(builder, GPUCompiler.Runtime.get(:signal_exception))
            unreachable!(builder)

            position!(builder, signal_match)
            ret!(builder)
        end

        call_function(llvm_f, Nothing, Tuple{UInt64,Int64}, :((signal,value)))
    end
end
@inline @generated function device_signal_wait_cas!(signal::UInt64, expected::Int64, value::Int64)
    JuliaContext() do ctx
        T_nothing = convert(LLVMType, Nothing, ctx)
        T_i32 = LLVM.Int32Type(ctx)
        T_i64 = LLVM.Int64Type(ctx)

        # create a function
        llvm_f, _ = create_function(T_nothing, [T_i64, T_i64, T_i64])
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            signal_match = BasicBlock(llvm_f, "signal_match", ctx)
            signal_miss = BasicBlock(llvm_f, "signal_miss", ctx)
            signal_miss_1 = BasicBlock(llvm_f, "signal_miss_1", ctx)
            signal_miss_2 = BasicBlock(llvm_f, "signal_miss_2", ctx)
            signal_fail = BasicBlock(llvm_f, "signal_fail", ctx)

            position!(builder, entry)
            br!(builder, signal_miss)

            position!(builder, signal_miss)
            T_sleep = LLVM.FunctionType(T_nothing, [T_i32])
            sleep_f = LLVM.Function(mod, "llvm.amdgcn.s.sleep", T_sleep)
            call!(builder, sleep_f, [ConstantInt(Int32(1), ctx)])
            T_signal_cas = LLVM.FunctionType(T_i64, [T_i64, T_i64, T_i64, T_i32])
            signal_cas = LLVM.Function(mod, "__ockl_hsa_signal_cas", T_signal_cas)
            loaded_value = call!(builder, signal_cas, [parameters(llvm_f)[1],
                                                       parameters(llvm_f)[2],
                                                       parameters(llvm_f)[3],
                                                       # __ATOMIC_ACQREL == 4
                                                       ConstantInt(Int32(4), ctx)])
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, parameters(llvm_f)[2])
            br!(builder, cond, signal_match, signal_miss_1)

            position!(builder, signal_miss_1)
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, ConstantInt(Int64(DEVICE_ERR_SENTINEL), ctx))
            br!(builder, cond, signal_fail, signal_miss_2)

            position!(builder, signal_miss_2)
            cond = icmp!(builder, LLVM.API.LLVMIntEQ, loaded_value, ConstantInt(Int64(HOST_ERR_SENTINEL), ctx))
            br!(builder, cond, signal_fail, signal_miss)

            position!(builder, signal_fail)
            call!(builder, GPUCompiler.Runtime.get(:signal_exception))
            unreachable!(builder)

            position!(builder, signal_match)
            ret!(builder)
        end

        call_function(llvm_f, Nothing, Tuple{UInt64,Int64,Int64}, :((signal,expected,value)))
    end
end
"Calls the host function stored in `hc` with arguments `args`."
@inline function hostcall!(hc::HostCall, args...)
    _hostcall_lock!(hc)
    _hostcall_write_args!(hc, args...)
    _hostcall!(hc)
end
@generated function _hostcall_lock!(hc::HostCall{UInt64,RT,AT}) where {RT,AT}
    # Acquire lock on hostcall buffer
    :($device_signal_wait_cas!(hc.signal, $READY_SENTINEL, $DEVICE_LOCK_SENTINEL))
end
@generated function _hostcall_write_args!(hc::HostCall{UInt64,RT,AT}, args...) where {RT,AT}
    ex = Expr(:block)

    # Copy arguments into buffer
    # Modified from CUDAnative src/device/cuda/dynamic_parallelism.jl
    if write_args
        off = 1
        for i in 1:length(args)
            T = args[i]
            sz = sizeof(T)
            # TODO: Should we do what CUDAnative does instead?
            ptr = :(Base.unsafe_convert(DevicePtr{$T,AS.Global}, hc.buf_ptr+$off-1))
            push!(ex.args, :(Base.unsafe_store!($ptr, args[$i])))
            off += sz
        end
    end

    ex
end
struct HostCallDeviceException end
@generated function _hostcall!(hc::HostCall{UInt64,RT,AT}) where {RT,AT}
    ex = Expr(:block)

    # Ring the doorbell
    push!(ex.args, quote
        $memfence!(Val(:seq_cst))
        # FIXME: $device_signal_wait_cas!(hc.signal, $DEVICE_LOCK_SENTINEL, $DEVICE_MSG_SENTINEL)
        $device_signal_store!(hc.signal, $DEVICE_MSG_SENTINEL)
    end)

    @gensym result ptr
    if RT === Nothing
        # Async hostcall, host will set READY_SENTINEL
        push!(ex.args, :($result = nothing))
    else
        push!(ex.args, quote
            # Wait on host message
            $device_signal_wait(hc.signal, $HOST_MSG_SENTINEL)
            # Get return buffer and load first value
            $memfence!(Val(:seq_cst))
            $ptr = Base.unsafe_convert(DevicePtr{DevicePtr{$RT,AS.Global},AS.Global}, hc.buf_ptr)
            $ptr = unsafe_load($ptr)
            if UInt64($ptr) == 0
                $device_signal_store!(hc.signal, $DEVICE_ERR_SENTINEL)
                AMDGPU.signal_exception()
            end
            $result = unsafe_load($ptr)::$RT
            $memfence!(Val(:seq_cst))
            $device_signal_store!(hc.signal, $READY_SENTINEL)
        end)
    end
    push!(ex.args, :($result))

    ex
end

## hostcall

@generated function _hostcall_args(hc::HostCall{UInt64,RT,AT}) where {RT,AT}
    ex = Expr(:tuple)

    # Copy arguments into buffer
    off = 1
    for i in 1:length(AT.parameters)
        T = AT.parameters[i]
        sz = sizeof(T)
        # TODO: Should we do what CUDAnative does instead?
        push!(ex.args, quote
            lref = Ref{$T}()
            HSA.memory_copy(convert(Ptr{Cvoid}, Base.unsafe_convert(Ptr{$T}, lref)),
                            convert(Ptr{Cvoid}, Base.unsafe_convert(Ptr{UInt8}, hc.buf_ptr+$(off-1))),
                            $sz)
            lref[]
        end)
        off += sz
    end

    return ex
end

struct HostCallException <: Exception
    reason::String
    err::Union{Exception,Nothing}
    bt::Union{Vector,Nothing}
end
HostCallException(reason) = HostCallException(reason, nothing, nothing)
HostCallException(reason, err) = HostCallException(reason, err, catch_backtrace())
function Base.showerror(io::IO, err::HostCallException)
    print(io, "HostCallException")
    if err.err !== nothing
        print(io, ":\n")
        Base.showerror(io, err.err)
        Base.show_backtrace(io, err.bt)
    end
end

"""
    HostCall(func, return_type::Type, arg_types::Type{Tuple}) -> HostCall

Construct a `HostCall` that executes `func` with the arguments passed from the
calling kernel. `func` must be passed arguments of types contained in
`arg_types`, and must return a value of type `return_type`, or else the
hostcall will fail with undefined behavior.

Note: This API is currently experimental and is subject to change at any time.
"""
function HostCall(func, rettype, argtypes; return_task=false,
                  agent=get_default_agent(), maxlat=DEFAULT_HOSTCALL_LATENCY,
                  continuous=false, buf_size=nothing)
    signal = HSASignal()
    hc = HostCall(rettype, argtypes, signal.signal[].handle; agent=agent, buf_size=buf_size)

    tsk = @async begin
        ret_buf = Ref{Mem.Buffer}()
        ret_len = 0
        try
            while true
                try
                    if !_hostwait(signal.signal[]; maxlat=maxlat)
                        throw(HostCallException("Hostcall: Device error on signal $(signal.signal[])"))
                    end
                catch err
                    throw(HostCallException("Hostcall: Error during hostwait", err))
                end
                # FIXME: Lock the signal
                if length(argtypes.parameters) > 0
                    args = try
                        _hostcall_args(hc)
                    catch err
                        throw(HostCallException("Hostcall: Error getting arguments", err))
                    end
                    @debug "Hostcall: Got arguments of length $(length(args))"
                else
                    args = ()
                end
                ret = try
                    func(args...,)
                catch err
                    throw(HostCallException("Hostcall: Error executing host function", err))
                end
                if rettype === Nothing
                    HSA.signal_store_release(signal.signal[], READY_SENTINEL)
                    continue
                end
                if typeof(ret) != rettype
                    throw(HostCallException("Hostcall: Host function result of wrong type: $(typeof(ret)), expected $rettype"))
                end
                if !isbits(ret)
                    throw(HostCallException("Hostcall: Host function result not isbits: $(typeof(ret))"))
                end
                @debug "Hostcall: Host function returning value of type $(typeof(ret))"
                try
                    if isassigned(ret_buf) && (ret_len != sizeof(ret))
                        Mem.free(ret_buf[])
                        ret_len = sizeof(ret)
                        ret_buf[] = Mem.alloc(agent, ret_len)
                    elseif !isassigned(ret_buf)
                        ret_len = sizeof(ret)
                        ret_buf[] = Mem.alloc(agent, ret_len)
                    end
                    ret_ref = Ref{UInt64}(ret)
                    GC.@preserve ret_ref begin
                        ret_ptr = convert(Ptr{Cvoid}, Base.unsafe_convert(Ptr{UInt64}, ret_ref))
                        HSA.memory_copy(Base.unsafe_convert(Ptr{Cvoid}, ret_buf[]), ret_ptr, sizeof(ret))
                    end
                    args_buf_ptr = reinterpret(Ptr{Cvoid}, hc.buf_ptr)
                    HSA.memory_copy(args_buf_ptr, Base.unsafe_convert(Ptr{Cvoid}, ret_buf[]), sizeof(UInt64))
                    HSA.signal_store_release(signal.signal[], HOST_MSG_SENTINEL)
                catch err
                    throw(HostCallException("Hostcall: Error returning hostcall result", err))
                end
                @debug "Hostcall: Host function return completed"
                continuous || break
            end
        catch err
            Base.showerror(stderr, err)
            Base.show_backtrace(stderr, catch_backtrace())
            rethrow(err)
        finally
            if isassigned(ret_buf)
                Mem.free(ret_buf[])
            end
            HSA.signal_store_release(signal.signal[], HOST_ERR_SENTINEL)
            Mem.free(hc.buf_ptr)
        end
    end

    if return_task
        return hc, tsk
    else
        return hc
    end
end

# CPU functions
get_value(hc::HostCall{UInt64,RT,AT} where {RT,AT}) =
    HSA.signal_load_scacquire(HSA.Signal(hc.signal))
function _hostwait(signal; maxlat=DEFAULT_HOSTCALL_LATENCY)
    @debug "Hostcall: Waiting on signal $signal"
    while true
        prev = HSA.signal_cas_scacq_screl(signal, DEVICE_MSG_SENTINEL, HOST_LOCK_SENTINEL)
        if prev == DEVICE_MSG_SENTINEL
            @debug "Hostcall: Device message on signal $signal"
            return true
        elseif prev == DEVICE_ERR_SENTINEL
            @debug "Hostcall: Device error on signal $signal"
            return false
        else
            @debug "Hostcall: Value of $prev on signal $signal"
            sleep(0.1)
        end
        sleep(maxlat)
    end
end
