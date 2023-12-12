Do the following:

1. Run the `mnist_train.kojo` script. This will train a model to recognize digits, and save it in a file called `mnist.djl.model`
2. Then run the `mnist_test.kojo` script. This will load `mnist.djl.model`, and use this model/neural-net to make predictions for the digit images stored under the `test` directory
3. Next, run the `mnist_test_draw.kojo` script. This will load `mnist.djl.model`, and then let you draw digits in the drawing canvas, for which it will make predictions based on the loaded neural-net

After you have all of this running, pause for a moment and think about how you have been able to tain a simple neural network in less than a minute to solve a complex problem like digit recognition in an image! 

Now, proceed to do the following experiments:

### Changes to mnist_train.kojo
For each of the following ideas, after you make the change (to `mnist_train.kojo`), retrain/save the neural network, and then run `mnist_test.kojo` (and also `mnist_test_draw.kojo` if you want to):
1. Change the random seed
2. Change the number of epochs of training
3. Change the number of hidden units in the network
 
