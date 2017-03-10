package com.knoldus

import breeze.linalg.{norm, DenseVector => BreeezeDenseVector}
import com.knoldus.Conversions._
import org.apache.spark.mllib.linalg.{DenseVector => SparlMlLibDenseVector}
import math._

object Formulas {

  /*
  Consine Similarity measures the cosine of the angle between the two none-zero vectors
  @return Double
  @param  two non-zero vectors
   */
  def computeCosineSimilarity(vectorOne: BreeezeDenseVector[Double], vectorTwo: BreeezeDenseVector[Double]): Double = {

    (vectorOne dot vectorTwo) / (norm(vectorOne) * norm(vectorTwo))
  }

  /*
  Consine Similarity measures the cosine of the angle between the two none-zero vectors
  @return Double
  @param  two non-zero array of type double
   */
  def computeCosineSimilarity(arrayOne: Array[Double], arrayTwo: Array[Double]): Double = {

    val vectorOne = parseArrayToBreezeDenseVector(arrayOne)
    val vectorTwo = parseArrayToBreezeDenseVector(arrayTwo)

    computeCosineSimilarity(vectorOne, vectorTwo)
  }

  /*
  Consine Similarity measures the cosine of the angle between the two none-zero vectors
  @return Double
  @param  two non-zero vectors
   */
  def computeCosineSimilarity(vectorOne: SparlMlLibDenseVector, vectorTwo: SparlMlLibDenseVector): Double = {

    val breezeVectorOne = parseMlLibDenseVecToBreeze(vectorOne)
    val breezeVectorTwo = parseMlLibDenseVecToBreeze(vectorTwo)

    computeCosineSimilarity(breezeVectorOne, breezeVectorTwo)
  }

  /*
  Dot Product is the angle between vectors A and B.
  @return Double
  @param two non-zero array of Double
   */
  def dotProduct(arrayOne: Array[Double], arrayTwo: Array[Double]): Double = {

    ((arrayOne zip arrayTwo).map { case (x, y) => x * y } :\ 0.0) (_ + _)
  }

  /*
  Dot Product is the angle between vectors A and B.
  @return Double
  @param two non-zero vectors
   */
  def dotProduct(vectorOne: BreeezeDenseVector[Double], vectorTwo: BreeezeDenseVector[Double]): Double = vectorOne dot vectorTwo

  /*
  Dot Product is the angle between vectors A and B.
  @return Double
  @param two non-zero vectors
   */
  def dotProduct(vectorOne: SparlMlLibDenseVector, vectorTwo: SparlMlLibDenseVector): Double = {

    val breezeVectorOne = parseMlLibDenseVecToBreeze(vectorOne)
    val breezeVectorTwo = parseMlLibDenseVecToBreeze(vectorTwo)

    dotProduct(breezeVectorOne, breezeVectorTwo)
  }

  /*
  Distance between vectors defines how far vector X is from vector Y in Vector Space.
  This is also known as Distinctiveness between vector or coupling between vectors
   @return Double
   @param two non-zero array of Double
   */
  def distanceBetweenVectors(arrayOne: Array[Double], arrayTwo: Array[Double]): Double = {

    sqrt((arrayOne zip arrayTwo).map { case (x, y) => pow(y - x, 2) }.sum)
  }

  /*
  Distance between vectors defines how far vector X is from vector Y in Vector Space.
  This is also known as Distinctiveness between vector or coupling between vectors
   @return Double
   @param two non-zero vectore
   */
  def distanceBetweenVectors(vectorOne: BreeezeDenseVector[Double], vectorTwo: BreeezeDenseVector[Double]): Double = {

    val arrayOne = vectorOne.toArray
    val arrayTwo = vectorTwo.toArray

    distanceBetweenVectors(arrayOne, arrayTwo)
  }

  /*
  Distance between vectors defines how far vector X is from vector Y in Vector Space.
  This is also known as Distinctiveness between vector or coupling between vectors
   @return Double
   @param two non-zero vectore
   */
  def distanceBetweenVectors(vectorOne: SparlMlLibDenseVector, vectorTwo: SparlMlLibDenseVector): Double = {

    val arrayOne = vectorOne.toArray
    val arrayTwo = vectorTwo.toArray

    distanceBetweenVectors(arrayOne, arrayTwo)
  }

  /*
  Mean is average of a particular list
  @return Double
  @param list of Double
   */
  def getMean(list: List[Double]): Double = {

    list match {
      case Nil => 0.0
      case noneEmptyList => noneEmptyList.sum / noneEmptyList.length.toDouble
    }
  }

  /*
  Mean is average of a particular list
  @return Double
  @param array of Double
   */
  def getMean(array: Array[Double]): Double = {

    val list = array.toList
    getMean(list)
  }

  /*
  Mean is average of a particular list
  @return Double
  @param vector
   */
  def getMean(vector: SparlMlLibDenseVector): Double = {

    val list = vector.toArray.toList
    getMean(list)
  }

  /*
  Mean is average of a particular list
  @return Double
  @param vector
   */
  def getMean(vector: BreeezeDenseVector[Double]): Double = {

    val list = vector.toArray.toList
    getMean(list)
  }

  /*
  Standard Deviation is a quantity expressing by how much the members of a group differ from the mean value for the group.
  @return Double
  @param list of Double
  @param mean of List
   */
  def getStandardDeviation(list: List[Double], mean: Double): Double = {

    list match {
      case Nil => 0.0
      case nonEmptyList => sqrt((0.0 /: nonEmptyList) { (x, y) =>
        x + pow(y - mean, 2.0)
      } / nonEmptyList.length)
    }
  }

  /*
  Standard Deviation is a quantity expressing by how much the members of a group differ from the mean value for the group.
  @return Double
  @param array of Double
  @param mean of List
   */
  def getStandardDeviation(array: Array[Double], mean: Double): Double = {

    val list = array.toList
    getStandardDeviation(list, mean)
  }

  /*
  Standard Deviation is a quantity expressing by how much the members of a group differ from the mean value for the group.
  @return Double
  @param vector
   */
  def getStandardDeviation(vector: SparlMlLibDenseVector): Double = {

    val mean = getMean(vector)
    val list = vector.toArray

    getStandardDeviation(list, mean)
  }

  /*
  Standard Deviation is a quantity expressing by how much the members of a group differ from the mean value for the group.
  @return Double
  @param vector
  @param mean of vector
   */
  def getStandardDeviation(vector: BreeezeDenseVector[Double]): Double = {

    val mean = getMean(vector)
    val list = vector.toArray

    getStandardDeviation(list, mean)
  }

  /*
  Cohesiveness is the distance between the elements of a cluster. the lower the distance between elements the better the cluster.
  can you evaluate weather the vector is Dense or not
  @return List of (clusterElement, differenceClusterElement, distance between these two elements)
  @param Array of clusterItems and there vector value as Array
   */
  def getCohesiveness[T](cluster: Array[(T, Array[Double])]): List[(T, T, Double)] = {

    val termsOfCluster = cluster.map(_._1)
    val allCombinationOfTerms = termsOfCluster.combinations(2).map { case Array(x, y) => (x, y) }.toList

    allCombinationOfTerms.map { case (termOne, termTwo) =>
      val termOneVector = cluster.filter(_._1 == termOne).map(_._2).take(1)(0)
      val termTwoVector = cluster.filter(_._1 == termTwo).map(_._2).take(1)(0)
      val distanceMeasure = distanceBetweenVectors(termOneVector, termTwoVector)
      (termOne, termTwo, distanceMeasure)
    }
  }

  /*
  Cohesiveness is the distance between the elements of a cluster. the lower the distance between elements the better the cluster.
  can you evaluate weather the vector is Dense or not
  @return List of (clusterElement, differenceClusterElement, distance between these two elements)
  @param Array of clusterItems and there vector value
   */
  def getCohesiveness[T](cluster: Array[(T, SparlMlLibDenseVector)]): List[(T, T, Double)] = {

    val mappedCluster = cluster.map { case (termOfCluster, vector) => (termOfCluster, vector.toArray) }
    getCohesiveness(mappedCluster)
  }

  /*
  Cohesiveness is the distance between the elements of a cluster. the lower the distance between elements the better the cluster.
  can you evaluate weather the vector is Dense or not
  @return List of (clusterElement, differenceClusterElement, distance between these two elements)
  @param Array of clusterItems and there vector value
   */
  def getCohesiveness[T](cluster: Array[(T, BreeezeDenseVector[Double])]): List[(T, T, Double)] = {

    val mappedCluster = cluster.map { case (termOfCluster, vector) => (termOfCluster, vector.toArray) }
    getCohesiveness(mappedCluster)
  }

}
