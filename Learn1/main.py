import numpy as np
import matplotlib.pyplot as plt
import utils

images, lables = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epoch = 3
e_loss = 0
e_correct = 0
learning_rate = 0.03

for epoch in range(epoch):
    print(f"Epoch: {epoch}")

    for image, lable in zip(images, lables):
        image = np.reshape(image, (-1, 1))
        lable = np.reshape(lable, (-1, 1))

        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))

        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1+np.exp(-output_raw))

        #loss
        e_loss += 1/len(output) * np.sum((output-lable)**2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(lable))

        #Backpropagation
        delta_output = output - lable
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

print(f"Loss: {round((e_loss[0] / images.shape[0]), 3)}%")
print(f"Accuracy: {round((e_correct / images.shape[0]), 3)}%")
e_loss = 0
e_correct = 0
print(output)

test_image = plt.imread("custom.jpg", format="jpeg")
gray = lambda rgb : np.dot(rgb[... , :3], [0.299, 0.587, 0.114])
test_image = 1 - (gray(test_image).astype("float32") / 255)
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

#TEST
plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"AI think this is: {output.argmax()}")
plt.show()