import numpy as np
from keras import layers
from keras.models import Sequential
from metaflow import FlowSpec, step
from sklearn.model_selection import train_test_split


def _func_to_learn(x: int) -> int:
    return (x % 69) + 420


class LearnModSpec(FlowSpec):
    """
    Learn to take modulo 69 and subtract 420
    """

    @step
    def start(self):
        print("Starting and generating data")

        self.inputs = np.array(list(range(0, 1_000_000)))
        self.outputs = np.array(list(map(_func_to_learn, self.inputs)))

        self.next(self.split)

    @step
    def split(self):
        print("Splitting data into train and test sets")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.outputs,
                                                                                test_size=0.5, random_state=42)
        self.next(self.train)

    @step
    def train(self):
        # TODO: Parallelize this step
        print("Training model")

        model = Sequential([
            layers.Dense(512, activation='relu', input_shape=(1,)),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(self.x_train, self.y_train, epochs=100, verbose=2, batch_size=10_000,
                  validation_data=(self.x_test, self.y_test))

        model.save("output/model.h5")

        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    LearnModSpec()
