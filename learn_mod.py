from metaflow import FlowSpec, step, Parameter, retry, kubernetes


def _func_to_learn(x: int) -> int:
    return (x % 69) + 420


class LearnModSpec(FlowSpec):
    """
    Learn the target function
    """

    dimensionality = Parameter('dimensionality', help='Dimensionality of the input data', default=512)
    input_size = Parameter('input_size', help='Size of the input data', default=1_000_000)
    epochs = Parameter('epochs', help='Number of epochs to train for', default=100)
    verbosity = Parameter('verbosity', help='Verbosity of training', default=2)
    batch_size = Parameter('batch_size', help='Batch size for training', default=10_000)

    @step
    def start(self):
        import numpy as np
        print("Starting and generating data")

        self.inputs = np.array(list(range(0, self.input_size)))
        self.outputs = np.array(list(map(_func_to_learn, self.inputs)))

        self.next(self.split)

    @step
    def split(self):
        from sklearn.model_selection import train_test_split
        print("Splitting data into train and test sets")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.outputs,
                                                                                test_size=0.5, random_state=42)
        self.next(self.train)

    @kubernetes(cpu=1, memory=500, image='tensorflow/tensorflow')
    @retry
    @step
    def train(self):
        from keras.models import Sequential
        from keras import layers

        print("Training the model")

        model = Sequential([
            layers.Dense(self.dimensionality, activation='relu', input_shape=(1,)),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(self.x_train, self.y_train, epochs=self.epochs, verbose=self.verbosity, batch_size=self.batch_size,
                  validation_data=(self.x_test, self.y_test))

        # Save to a predefined location for ease of access in local runs
        # Persist the bytes as a metaflow artifact: https://outerbounds.com/docs/use-keras-with-metaflow/
        model.save("output/model.h5")
        with open("output/model.h5", "rb") as f:
            self.model = f.read()

        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    LearnModSpec()
