using JITrench

macro nograd(ex)
    JITrench.nograd = true
    res = Core.eval(__module__, ex)
    JITrench.nograd = false
    return res
end

