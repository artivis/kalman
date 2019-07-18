// The MIT License (MIT)
//
// Copyright (c) 2015 Markus Herb
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef KALMAN_TYPES_HPP_
#define KALMAN_TYPES_HPP_

#include "Matrix.hpp"

namespace Kalman
{
namespace internal
{

    /**
     * @class Kalman::internal::traits
     * @brief
     */
    template<typename T>
    struct traits
    {
      using Scalar = typename T::Scalar;
      static constexpr auto Size = T::RowsAtCompileTime;
    };

}


    /**
     * @class Kalman::SquareMatrix
     * @brief Template type representing a square matrix
     * @param T The numeric scalar type
     * @param N The dimensionality of the Matrix
     */
    template<typename T, int N>
    using SquareMatrix = Matrix<T, N, N>;

    /**
     * @class Kalman::Covariance
     * @brief Template type for covariance matrices
     * @param Type The vector type for which to generate a covariance (usually a state or measurement type)
     */
    template<class Type>
    using Covariance = SquareMatrix<typename internal::traits<Type>::Scalar, internal::traits<Type>::Size>;

    /**
     * @class Kalman::CovarianceSquareRoot
     * @brief Template type for covariance square roots
     * @param Type The vector type for which to generate a covariance (usually a state or measurement type)
     */
    template<class Type>
    using CovarianceSquareRoot = Cholesky< Covariance<Type> >;

    /**
     * @class Kalman::KalmanGain
     * @brief Template type of Kalman Gain
     * @param State The system state type
     * @param Measurement The measurement type
     */
    template<class State, class Measurement>
    using KalmanGain = Matrix<typename internal::traits<State>::Scalar,
                              internal::traits<State>::Size,
                              Measurement::RowsAtCompileTime>;

    /**
     * @class Kalman::Jacobian
     * @brief Template type of jacobian of A w.r.t. B
     */
    template<class A, class B>
    using Jacobian = Matrix<typename internal::traits<A>::Scalar,
                            internal::traits<A>::Size,
                            internal::traits<B>::Size>;

    template <typename T, int N, int RC>
    bool isSymmetric(const Eigen::Matrix<T, N, N, RC>& M,
                     const T eps = 1e-8)
    {
      return M.isApprox(M.transpose(), eps);
    }

    template <typename T, int N, int RC>
    bool isPositiveSemiDefinite(const Eigen::Matrix<T, N, N, RC>& M,
                                const T eps = 1e-8)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N, RC> > eigensolver(M);

      if (eigensolver.info() == Eigen::Success)
      {
        // All eigenvalues must be >= 0:
        return (eigensolver.eigenvalues().array() >= eps).all();
      }

      return false;
    }

    template <typename T, int N, int RC>
    bool isCovariance(const Eigen::Matrix<T, N, N, RC>& M,
                      const T eps = 1e-8)
    {
      return isSymmetric(M) && isPositiveSemiDefinite(M, eps);
    }

    template <typename T, int N, int RC>
    bool makePosDef(Eigen::Matrix<T,N,N,RC>& M, const T eps = 1e-8)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N,RC> > eigensolver(M);

        if (eigensolver.info() == Eigen::Success)
        {
            // All eigenvalues must be >= 0:
            T epsilon = eps;
            while ((eigensolver.eigenvalues().array() < eps).any())
            {
                //std::cout << "----- any negative eigenvalue or too close to zero\n";
                //std::cout << "previous eigenvalues: " << eigensolver.eigenvalues().transpose() << std::endl;
                //std::cout << "previous determinant: " << M.determinant() << std::endl;
                M = eigensolver.eigenvectors() *
                    eigensolver.eigenvalues().cwiseMax(epsilon).asDiagonal() *
                    eigensolver.eigenvectors().transpose();
                eigensolver.compute(M);
                //std::cout << "epsilon used: " << epsilon << std::endl;
                //std::cout << "posterior eigenvalues: " << eigensolver.eigenvalues().transpose() << std::endl;
                //std::cout << "posterior determinant: " << M.determinant() << std::endl;
                epsilon *= 10;
            }

            if (!isCovariance(M))
              throw std::runtime_error("Matrix is not a covariance!");

            return epsilon != eps;
        }
        else
            throw std::runtime_error("SelfAdjointEigenSolver failed!");

        return false;
    }


}

#endif
