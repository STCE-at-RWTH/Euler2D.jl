function between(ival, x::T) where {T}
    proper = (ival[1] < ival[2])
    if proper && (ival[1] < x < ival[2])
        return one(T)
    elseif !proper && (ival[2] < x < ival[1])
        return -one(T)
    else
        return zero(T)
    end
end
@inline X_in(ival) = Base.Fix1(between, ival)