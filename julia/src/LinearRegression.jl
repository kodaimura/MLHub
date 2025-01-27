import Pkg
Pkg.activate(".")

using Flux
using Plots

function loss_mse(model, features, labels)
    y_hat = model(features)
    return Flux.mse(y_hat, labels)
end

function loss_lasso(model, features, labels; lambda=0.01)
    y_hat = model(features)
    mse_loss = Flux.mse(y_hat, labels)
    l1_penalty = lambda * sum(abs.(model.weight))
    return mse_loss + l1_penalty
end

function loss_ridge(model, features, labels; lambda=0.01)
    y_hat = model(features)
    mse_loss = Flux.mse(y_hat, labels)
    l2_penalty = lambda * sum(model.weight .^ 2)
    return mse_loss + l2_penalty
end

function loss_elastic_net(model, features, labels; lambda1=0.01, lambda2=0.01)
    y_hat = model(features)
    mse_loss = Flux.mse(y_hat, labels)
    l1_penalty = lambda1 * sum(abs.(model.weight))
    l2_penalty = lambda2 * sum(model.weight .^ 2)
    return mse_loss + l1_penalty + l2_penalty
end

function train_model!(loss, model, features, labels; learning_rate=0.01)
    dLdm, _, _ = gradient(loss, model, features, labels)
    @. model.weight = model.weight - learning_rate * dLdm.weight
    @. model.bias = model.bias - learning_rate * dLdm.bias
end

function train_model!(loss, model, data; learning_rate=0.01)
    Flux.train!(loss, model, data, Descent(learning_rate))
end

function train_until_converged!(loss, model, data; max_epochs=10000, tolerance=1e-4, learning_rate=0.01)
    x = hcat([d[1] for d in data]...)
    y = hcat([d[2] for d in data]...)
    loss_prev = Inf
    for epoch in 1:max_epochs
        train_model!(loss, model, data; learning_rate=learning_rate)
        current_loss = loss(model, x, y)

        if loss_prev == Inf
            loss_prev = current_loss
            continue
        end

        if current_loss < 1 && abs(loss_prev - current_loss) < tolerance
            println("Converged at epoch $epoch with loss $current_loss")
            break
        end
        loss_prev = current_loss
    end
end

function main()
    #単回帰サンプル
    (function()
        data = [([x + rand(Float32)], 3x + 5 + rand(Float32)) for x in -3:0.1f0:3]
        x = hcat([d[1] for d in data]...)
        y = hcat([d[2] for d in data]...)

        loss = (model, x, y) -> loss_mse(model, x, y)
        model = Flux.Dense(1 => 1)
        train_until_converged!(loss, model, data; tolerance=1e-6)
        predicted_values = model(x)

        plot(vec(x), vec(y), seriestype = :scatter, label="True values", title="Model Training")
        plot!((x) -> model.bias[1] + model.weight[1] * x, label="After Training", lw=2)

        savefig("LinearRegression.png")
    end)()

    #重回帰サンプル
    (function()
        f(x) = 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1
        data = []
        for i in 1:100
            xi = [rand(Float32), rand(Float32), rand(Float32), rand(Float32), rand(Float32)]
            yi = f(xi) + rand(Float32)
            push!(data, (xi, yi))
        end
        x = hcat([d[1] for d in data]...)
        y = hcat([d[2] for d in data]...)
        
        loss = (model, x, y) -> loss_mse(model, x, y)
        model = Flux.Dense(5 => 1)
        train_until_converged!(loss, model, data; tolerance=1e-6)
        predicted_values = model(x)

        plot(1:100, y[1, :], seriestype = :scatter, label="True values", title="Model Training")
        plot!(1:100, predicted_values[1, :], label="After Training")

        savefig("LinearRegression2.png")
    end)()
end

main()