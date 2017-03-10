package com.knoldus

import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SparlMlLibDenseVector}
import breeze.linalg.{norm, DenseVector => BreeezeDenseVector}


object Conversions {

  def parseMlLibDenseVecToBreeze(vector: SparlMlLibDenseVector): BreeezeDenseVector[scala.Double] = BreeezeDenseVector(vector.toArray)

  def parseArrayToBreezeDenseVector(arrayOfDouble: Array[Double]): BreeezeDenseVector[scala.Double] = BreeezeDenseVector(arrayOfDouble)

  def parseBreezeDenseVectorToMlLibDenseVector(vector: BreeezeDenseVector[Double]): SparlMlLibDenseVector = Vectors.dense(vector.toArray).toDense

}
