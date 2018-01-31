// basically copied from: https://blog.webkid.io/neural-networks-in-javascript/

const mnist = require('mnist');
const synaptic = require('synaptic'); 

const set = mnist.set(700, 20)

const trainingSet = set.training
const testSet = set.test

const Layer = synaptic.Layer;
const Network = synaptic.Network;
const Trainer = synaptic.Trainer;

const inputLayer = new Layer(784);
const hiddenLayer1 = new Layer(100);
const hiddenLayer2 = new Layer(100); 
const outputLayer = new Layer(10);

inputLayer.project(hiddenLayer1); 
hiddenLayer1.project(hiddenLayer2, Layer.connectionType.ONE_TO_ONE);
hiddenLayer2.project(outputLayer);

const myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer1, hiddenLayer2],
    output: outputLayer
}); 

let trainer = new Trainer(myNetwork); 

trainer.train(trainingSet, {
  rate: .2,
  iterations: 2 ,
  error: .1,
  shuffle: false,
  log: 1,
  cost: Trainer.cost.CROSS_ENTROPY
})

console.log(myNetwork.activate(testSet[0].input))
console.log(testSet[0])