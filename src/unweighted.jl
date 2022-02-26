"""
UnweightedParticleFilter

A particle filter that does not use any reweighting, but only keeps particles if the observation matches the true observation exactly. This does not require obs_weight, but it will not work well in real-world situations.
"""
struct UnweightedParticleFilter{M, RNG<:AbstractRNG} <: Updater
    model::M
    n::Int
    rng::RNG
end

function UnweightedParticleFilter(model, n::Integer; rng=Base.GLOBAL_RNG)
    return UnweightedParticleFilter(model, n, rng)
end

function update(up::UnweightedParticleFilter, b::ParticleCollection, a, o)
    if n_particles(b) == 0
        bs = initialstate(up.model)
        tmp = Random.gentype(b)[]
        for i in 1:100
            push!(tmp, Random.rand(up.rng, bs))
        end
        b = ParticleCollection(tmp)
    end
    new = Random.gentype(b)[]
    i = 1
    while i <= up.n
        s = particle(b, mod1(i, n_particles(b)))  
        if isterminal(up.model, s)==false
            sp, o_gen = @gen(:sp, :o)(up.model, s, a, up.rng)
            if o_gen == o
                push!(new, sp)
            end
        end
        i += 1
    end
    if isempty(new)
        # @warn("""
        #      Particle Depletion!

        #      The UnweightedParticleFilter generated no particles consistent with observation $o. Consider upgrading to a BootstrapFilter or a BasicParticleFilter or creating your own domain-specific updater.
        #      """
        #     )
        i = 1
        while i <= up.n
            s = particle(b, mod1(i, n_particles(b)))
            if isterminal(up.model, s)==false
                sp, o_gen = @gen(:sp, :o)(up.model, s, a, up.rng)
                push!(new, sp)
            end
            i += 1
        end
    end
    if isempty(new)
        bs = initialstate(up.model)
        for i in 1:100
            push!(new, Random.rand(up.rng, bs))
        end
    end
    return ParticleCollection(new)
end 

function update(up::UnweightedParticleFilter, b, a, o)
    return update(up, initialize_belief(up, b), a, o)
end

function initialize_belief(up::UnweightedParticleFilter, b)
    return ParticleCollection(collect(rand(up.rng, b) for i in 1:up.n))
end
