import * as tf from '@tensorflow/tfjs'
import {Tensor, Rank} from '@tensorflow/tfjs'

const {mul, add, sum, div, sub, matMul, log, neg} = tf;

/**
 * From-scratch implementation of the Logistic Regression
 */
export class KNearestNeighbors {
    // w: tf.Scalar
    // b: tf.Scalar
    // gradients: {dw: Tensor<tf.Rank>; db: tf.Scalar}

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
     * Back propogate the updates
     * @param x - The input dataset
     * @param y - Actual labels
     */
    fit(x: Tensor, y: Tensor):void {
        this.selfFitX = x
        this.selfFitY = y
    }

    /**
     * Back propogate the updates
     * @param x - The input dataset
     * @returns The best match
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

