import Pkg
Pkg.activate(".")

using Flux
using MLDatasets 
using DataFrames
using Statistics

function loss(model, features, labels_onehot)
    y_hat = model(features)
    Flux.logitcrossentropy(y_hat, labels_onehot)
end

function train_model!(f_loss, model, features, labels_onehot; learning_rate=0.01)
    dLdm, _, _ = gradient(f_loss, model, features, labels_onehot)
    @. model[1].weight = model[1].weight - learning_rate * dLdm[:layers][1][:weight]
    @. model[1].bias = model[1].bias - learning_rate * dLdm[:layers][1][:bias]
end

function train_until_accuracy_reached!(f_loss, model, features, labels, classes; max_epochs=10000, threshold=0.98, learning_rate=0.01)
    labels_onehot = Flux.onehotbatch(labels, classes)
    accuracy(x, y) = Statistics.mean(Flux.onecold(model(x), classes) .== y)
    for epoch in 1:max_epochs
        train_model!(f_loss, model, features, labels_onehot; learning_rate=learning_rate)

        current_accuracy = accuracy(features, labels)
        if current_accuracy >= threshold
            println("Converged at epoch $epoch with accuracy $current_accuracy")
            break
        end
    end
end

function main()
    #サンプルデータ
    x, y = Iris(as_df=false)[:]
    x = Float32.(x)
    y = vec(y)
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

    model = Chain(Dense(4 => 3), softmax)
    features = x
    labels = y
    train_until_accuracy_reached!(loss, model, features, labels, classes; learning_rate=0.1)

    # 学習したモデルで予測
    new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [5.9, 3.0, 5.1, 1.8]]
    for sample in new_data
        x = Float32.(sample)
        y_hat = model(x)
        predicted_class = Flux.onecold(y_hat, classes)
        println("Predicted class for sample ", sample, ": ", predicted_class)
    end
end

main()