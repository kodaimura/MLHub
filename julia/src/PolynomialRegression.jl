import Pkg
Pkg.activate(".")

using Flux
using Plots

function loss(model, features, labels)
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

function train_model!(f_loss, model, features, labels; learning_rate=0.01)
    dLdm, _, _ = gradient(f_loss, model, features, labels)
    @. model.weight = model.weight - learning_rate * dLdm.weight
    @. model.bias = model.bias - learning_rate * dLdm.bias
end

function train_until_converged!(f_loss, model, features, labels; max_epochs=10000, tolerance=1e-4, learning_rate=0.01)
    loss_prev = Inf
    for epoch in 1:max_epochs
        train_model!(f_loss, model, features, labels; learning_rate=learning_rate)
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
    # 多項式回帰サンプル
    (function()
        # 入力データ
        x = collect(Float32, -3:0.1:3)
        f(x) = @. 2 + 3 * x + 5 * x^2 - 3 * x^3  # 実際の多項式
        y = reshape(f(x), 1, 61)
        y = y .+ reshape(rand(Float32, 61), (1, 61))
        x = map(a -> a + rand(Float32), x)

        degree = 3  # 多項式の次数
        poly_x = transpose(hcat([x .^ i for i in 1:degree]...))

        # モデルと損失関数を定義
        f_loss = (model, x, y) -> loss(model, x, y)
        model = Flux.Dense(degree => 1)

        # モデルをトレーニング
        train_until_converged!(f_loss, model, poly_x, y; learning_rate=0.001,  tolerance=1e-6)
        predicted_values = model(poly_x)

        # プロット
        plot(vec(x), vec(y), seriestype=:scatter, label="True values", title="Polynomial Regression")
        plot!((x) -> model.bias[1] + model.weight[3] * x^3 + model.weight[2] * x^2 + model.weight[1] * x, label="Predicted values", lw=2)
        savefig("PolynomialRegression.png")
    end)()
end

main()
