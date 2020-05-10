import * as tf from '@tensorflow/tfjs'
import {Tensor} from '@tensorflow/tfjs'

const {sub, neg} = tf;

/**
 * A simple implementation of KNN which will 
 * find the best neighbor combination required to yield the highest results
 * @constructor Instantiates a KNN Model
 * @returns An object instance of the model
 */
export class KNearestNeighbors {

    constructor(
        private numNeighbor: number,
        // private batchSize: number = 10,
        private selfFitX: Tensor,
        private selfFitY: Tensor,
    ) {
        const numClasses = new Set(selfFitY.dataSync()).size
        this.selfFitY = tf.oneHot(selfFitY.cast('int32'), numClasses)
    }

    /**
     * Using Euclidean distances, it finds for every sample in x where it should belong
     * based on the training data
     * @param x - The input dataset
     * @returns The predictions for where every sample from X belongs
     */
    predict(x: Tensor) {
        const distance = euclideanDistance(this.selfFitX,x)
        const {indices: topKIndices} = tf.topk(neg(distance), this.numNeighbor)

        const countPred = tf.gather(this.selfFitY, topKIndices).sum(1)
        const predictions = tf.argMax(countPred, 1)
        // console.log('Prediction Confidence: ', div(predCount, this.numNeighbor).dataSync())
        // return tf.argMax(predCount, 1)
        return predictions
    }
}

/**
 * Calculate the Euclidean Distance between two Tensors
 * @param x - Training Dataset
 * @param y - Test Dataset
 * @returns The distance between the two tensors
 */
function euclideanDistance(x: Tensor, y: Tensor) {
    // return tf.norm(, y), 'euclidean')
    return tf.norm(sub(x, y.expandDims(1)), undefined, 2)
}

