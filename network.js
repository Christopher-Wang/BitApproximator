var math = require("mathjs");
var data = require("./toy_data.json")
var adder = data.xor
//Test behavior, training to make a full adder


function train(network, epoch, check, learningRate){
	var size = network.layers - 1;
	var set = 0;
	for(var i = 0; i < epoch; i++){
		//Print the data
		if(i % check == 0){
			printTestdata(network, ++set);
		}
		//Do the training
		var trainingExample = Math.floor(Math.random() * Math.floor(adder.length));
		network.forwardPropagate(adder[trainingExample][0]);
		network.backProp(adder[trainingExample][1]);
		network.learn(learningRate);
	}
}

function printTestdata(network, set){
	console.log("Epoch:" + set);
	for(let j = 0; j < adder.length; j++){
		network.forwardPropagate(adder[j][0])
		console.log("Input: " + adder[j][0] + " Expected output: " + adder[j][1]);
		console.log("Cost: " + network.cost(adder[j][1]).toFixed(4) + " Actual output: " + network.activations[network.layers - 1]);
	}
	console.log("\n");
}

/** The constructor for a neuron. This is for the basic handwritten recognition test.
 * @param {number} numWeights The number of weights the neuron has
 */
function Neuron(numWeights){
	this.weights = [];
	for(let i = 0; i < numWeights; i++){
		this.weights.push(Math.random());
	}
	this.bias = Math.random();
}

/** Constructor for the network
 * @param {array} size An array containing the number of weights in each layer
 */
function Network(size){
	//Initialize all values here
	this.size = size;
	this.layers = size.length;
	this.weights = [];
	this.biases = [];
	this.weightedSum = [];
	this.activations = [];
	this.errors = [];
	this.dweights = [];

	for(let i = 1; i < this.layers; i++){//Start at 1 because first layer is input layer
		let weightMatrices = [];
		let biasVectors = [];
		for(let j = 0; j < size[i]; j++){//go through number of neurons in a layer
			let neuron = new Neuron(this.size[i-1]);
			weightMatrices.push(neuron.weights);
			biasVectors.push(neuron.bias);
		}
		this.weights.push(weightMatrices);//A 3d Matrix
		this.biases.push(biasVectors);
	}
}

/** Function for the network that determines the activations for the next input layer i.e. 
 * propagates forward.
 *	@param {array} inputLayer An array containing the activation values for the input laters
 */
Network.prototype.forwardPropagate = function(inputLayer){
	this.activations[0] = inputLayer;
	for(let i = 1; i < this.layers; i++){//Start at 1 because first layer is input layer
		this.activations[i] = this.activate(i-1);
	}
}

/** Function for the network that determines the activations for the next input layer i.e. 
 * propagates forward.
 *	@param {number} i The layer of neurons to be activated
 */
Network.prototype.activate = function(i){
	this.weightedSum[i] = math.add(math.multiply(this.weights[i], this.activations[i]), this.biases[i]);
	return this.weightedSum[i].map(sigmoid);
}

/**The sigmoid function
 * @param {number} sum The weighted sum
 * return {number} The output value
 */
function sigmoid(sum){
	return (1 / (1 + Math.exp(-sum)));
}

//double check this
/**The derivative of the sigmoid function
 * @param {number} sum The weighted sum
 * @return {number} The output value
 */
 function sigmoidPrime(sum){
 	return (Math.exp(-sum))/Math.pow((1+Math.exp(-sum)), 2);
 }

/**This is quadratic cost cost function 
 * @param {array} Expected output 
 * @return {number} The cost value
 */
 Network.prototype.cost = function(expected){
 	let sum = 0;
 	let lastLayer = this.layers - 1;
 	for (var i = 0; i < expected.length; i++){
 		sum += Math.pow((expected[i] - this.activations[lastLayer][i]), 2);
 	}
 	return sum / 2;
 }
//Time for the big daddy
 /** Calculate dC/dz for the output layer
  * @param {array} Expected output 
  */
Network.prototype.backProp = function(expected){
	//The total length of our error field is the same as the biases or weight
	var layer = this.biases.length;
	//The last layer is the length of the biases - 1
	this.errors[--layer] = this.firstFundamentalEquation(expected);
	//Propagate backwards through the network, starting from the second last layer
	for (--layer; layer >= 0; layer--) {
		this.errors[layer] = this.secondFundamentalEquation(layer);
	}
	for (var i = 0; i < this.biases.length; i++) {
		this.dweights[i] = this.thirdFundamentalEquation(i);
	}
}

/** Function to compute the Gradient of the last layer with respect to the weighted sum of last layer
 */
Network.prototype.firstFundamentalEquation = function (expected){
	var lastLayer = this.layers - 1;//This s the index I need to work with
	var costDerivative = math.subtract(expected, this.activations[lastLayer]);
	//Remember, the weighted sum is 1 size smaller than total number of layers
	var activationDerivative = this.weightedSum[lastLayer - 1].map(sigmoidPrime); 
	return math.dotMultiply(costDerivative, activationDerivative);
}

/** Function to compute the Gradient of a layer "layer" with respect to the weighted sum of layer
 * @param {number} layer The layer of interest
 *
 */
Network.prototype.secondFundamentalEquation = function(layer){
	var left = math.multiply(math.transpose(this.weights[layer + 1]), this.errors[layer+1]);
	var right = this.weightedSum[layer].map(sigmoidPrime);
	return math.dotMultiply(left, right);

}

/** Function that computes dC/dw
 * @param {number} layer The layer of interest (not this is the INPUT layer)
 * @return {matrix} A matrix contain the dC/dw of the layer
 */
Network.prototype.thirdFundamentalEquation = function(layer){
	//Remember that the indexes of weights and biases are off by 1 because
	//They have no input layer, so the 0th layer of activation corresponds to
	//the first set of weights which is the 0th layer
	return matrixBuilder(this.activations[layer], this.errors[layer]);
}

/** Function that implements Stochastic Gradient Descent 
 * @param {number} learningRate The learning rate of the network
 */
 Network.prototype.learn = function(learningRate){
 	for (var i = 0; i < this.biases.length; i++){
 		//What? I should be multiplying by the negative of the gradient, why do i use the positive
 		rate = learningRate;
 		//Update the biase
 		this.biases[i] = math.add(this.biases[i], math.multiply(this.errors[i], rate));
 		this.weights[i] = math.add(this.weights[i], math.multiply(this.dweights[i], rate));
 	}
 }

/** Method to construct a 2D array formed from the products of 2 arrays
 * 
 */
function matrixBuilder(arr1, arr2){
	var matrix = [];
	for(var i = 0; i < arr2.length; i++){
		var row = [];
		for(var j = 0; j < arr1.length; j++){
			row.push(arr2[i] * arr1[j]);
		}
		matrix.push(row);
	}
	return matrix;
}




var network = new Network([2, 2, 1]);
train(network, 100000, 10000, 3);
printTestdata(network, 11);
