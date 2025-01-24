import Pkg
Pkg.activate(".")

using Flux
using Plots

function loss(model, features, labels)
    y_hat = model(features)
    Flux.mse(y_hat, labels)
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

function train_model!(f_loss, model, features, labels, learning_rate=0.01)
    dLdm, _, _ = gradient(f_loss, model, features, labels)
    @. model.weight = model.weight - learning_rate * dLdm.weight
    @. model.bias = model.bias - learning_rate * dLdm.bias
end

function train_until_converged!(f_loss, model, features, labels; max_epochs=10000, tolerance=1e-4)
    loss_prev = Inf
    for epoch in 1:max_epochs
        train_model!(f_loss, model, features, labels)
        current_loss = f_loss(model, features, labels)

        if loss_prev == Inf
            loss_prev = current_loss
            continue
        end

        if abs(loss_prev - current_loss) < tolerance
            println("Converged at epoch $epoch with loss $current_loss")
            break
        end
        loss_prev = current_loss
    end
end

function main()
    #サンプルデータ
    x = rand(Float32, 5, 100)
    f(x) = @. 3 * x[1, :] + 2 * x[2, :] - x[3, :] + 4 * x[4, :] - 2 * x[5, :] + 1
    y = reshape(f(x), 1, 100)

    f_loss = (model, features, labels) -> loss_ridge(model, features, labels; lambda=0.05)
    model = Flux.Dense(5 => 1)
    features = x
    labels = y
    train_until_converged!(f_loss, model, features, labels)
    predicted_values = model(features)

    plot(1:100, labels[1, :], label="True values", lw=2, title="Model Training")
    plot!(1:100, predicted_values[1, :], label="After Training", lw=2)

    savefig("LinearRegression.png")
end

main()