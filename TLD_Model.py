from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, Input

def TLDModel(inputShape = (120, 60, 3), numClasses=3):
    
    inputLayer = Input(shape=inputShape)

    x = Conv2D(12, kernel_size= (5,5), strides=(2,2), padding="valid", activation="relu", name="TLD_Conv1")(inputLayer)
    x = Conv2D(24, kernel_size= (3,3), strides=(2,2), padding="valid", activation="relu", name="TLD_Conv2")(x)
    x = Conv2D(48, kernel_size= (3,3), strides=(2,2), padding="valid", activation="relu", name="TLD_Conv3")(x)

    x = Flatten(name="TLD_Flatten")(x)
    
    x = Dense(512, activation="relu", name="TLD_Dense1")(x)
    x = Dense(128, activation="relu", name="TLD_Dense2")(x)
    x = Dense(64, activation="relu", name="TLD_Dense3")(x)
    output = Dense(numClasses, activation="softmax", name="TLD_Output")(x)

    model = Model(inputs=inputLayer, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model

if __name__ == "__main__":
    model = TLDModel()
    model.summary()
