#ifndef KALMAN_EXAMPLES1_ROBOT_SYSTEMMODEL_HPP_
#define KALMAN_EXAMPLES1_ROBOT_SYSTEMMODEL_HPP_

#include <kalman/LinearizedSystemModel.hpp>

namespace KalmanExamples {
namespace CustomRobot1 {

/// We only declare this to make this
/// value meaning explicit.
///
/// It could be define in many other ways.
#define STATE_LOCAL_SIZE 3

/// Forward declaration
template<typename T>
class State;

} /* namespace CustomRobot1 */
} /* namespace KalmanExamples */

/// Some traits specializations for the custom class
///
/// template<typename T>
/// KalmanExamples::CustomRobot1::State<T>
///
namespace Kalman {

template<typename T>
struct CovarianceTrait<KalmanExamples::CustomRobot1::State<T>>
{
  using Covariance = SquareMatrix<T,STATE_LOCAL_SIZE>;
};

template<typename T, class B>
struct JacobianTrait<KalmanExamples::CustomRobot1::State<T>, B>
{
  using Jacobian = Matrix<T,STATE_LOCAL_SIZE,B::RowsAtCompileTime>;
};

template<class A, typename T>
struct JacobianTrait<A, KalmanExamples::CustomRobot1::State<T>>
{
  using Jacobian = Matrix<T,A::RowsAtCompileTime,STATE_LOCAL_SIZE>;
};

template<typename T>
struct JacobianTrait<KalmanExamples::CustomRobot1::State<T>,
                     KalmanExamples::CustomRobot1::State<T>>
{
  using Jacobian = Matrix<T,STATE_LOCAL_SIZE,STATE_LOCAL_SIZE>;
};

template<typename T, class Measurement>
struct KalmanGainTrait<KalmanExamples::CustomRobot1::State<T>, Measurement>
{
  using KalmanGain = Matrix<T,STATE_LOCAL_SIZE,Measurement::RowsAtCompileTime>;
};

template<typename T>
struct KalmanGainTrait<KalmanExamples::CustomRobot1::State<T>,
                       KalmanExamples::CustomRobot1::State<T>>
{
  using KalmanGain = Matrix<T,STATE_LOCAL_SIZE,STATE_LOCAL_SIZE>;
};

template <class T>
struct UpdateTrait<KalmanExamples::CustomRobot1::State<T>>
{
  using Update = Matrix<T,STATE_LOCAL_SIZE,1>;
};

template <class T>
struct MatrixTrait<KalmanExamples::CustomRobot1::State<T>>
{
  using SquareStateSizeMatrix = Matrix<T,STATE_LOCAL_SIZE,STATE_LOCAL_SIZE>;
};

} /* namespace Kalman */

namespace KalmanExamples {
namespace CustomRobot1 {

/**
 * @brief System state vector-type for a 3DOF planar robot.
 *
 * This is a system state for a very simple planar robot that
 * is characterized by the composite manifold R^2*S^1.
 *
 * @param T Numeric scalar type
 */
template<typename T>
class State
{
public:

    using Scalar  = T;
    using Tangent = Eigen::Matrix<T, 3, 1>;

    T x()     const { return x_; }
    T y()     const { return y_; }
    T theta() const { return std::arg(theta_); }

    void setZero()
    {
      x_     = 0;
      y_     = 0;
      theta_.real(1);
      theta_.imag(0);
    }

    State<T>& operator += (const Tangent& v)
    {
      x_ += v(0);
      y_ += v(1);
      auto exp_theta = std::complex<T>(std::cos(v(2)), std::sin(v(2)));
      theta_ *= exp_theta;
      return *this;
    }

    State<T> operator + (const Tangent& v) const
    {
      State<T> ret(*this);
      return ret += v;
    }

protected:

    T x_ = 0,
      y_ = 0;

    std::complex<T> theta_;
};

} /* namespace CustomRobot1 */
} /* namespace KalmanExamples */

namespace KalmanExamples {
namespace CustomRobot1 {

/**
 * @brief System control-input vector-type for a 3DOF planar robot
 *
 * This is the system control-input of a very simple planar robot that
 * can control the velocity in its current direction as well as the
 * change in direction.
 *
 * @param T Numeric scalar type
 */
template<typename T>
class Control : public Kalman::Vector<T, 2>
{
public:
    KALMAN_VECTOR(Control, T, 2)
    
    //! Velocity
    static constexpr size_t V = 0;
    //! Angular Rate (Orientation-change)
    static constexpr size_t DTHETA = 1;
    
    T v()       const { return (*this)[ V ]; }
    T dtheta()  const { return (*this)[ DTHETA ]; }
    
    T& v()      { return (*this)[ V ]; }
    T& dtheta() { return (*this)[ DTHETA ]; }
};

/**
 * @brief System model for a simple planar 3DOF robot
 *
 * This is the system model defining how our robot moves from one 
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
    typedef KalmanExamples::CustomRobot1::State<T> S;
    
    //! Control type shortcut definition
    typedef KalmanExamples::CustomRobot1::Control<T> C;
    
    /**
     * @brief Definition of (non-linear) state transition function
     *
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to 
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * @param [in] x The system state in current time-step
     * @param [in] u The control vector input
     * @returns The (predicted) system state in the next time-step
     */
    S f(const S& x, const C& u) const
    {       
        // New orientation given by old orientation plus orientation change
        T newOrientation = x.theta() + u.dtheta();
        // Re-scale orientation to [-pi/2 to +pi/2]
        
        // Return transitioned state vector
        //
        // New x-position given by old x-position plus change in x-direction
        // Change in x-direction is given by the cosine of the (new) orientation
        // times the velocity
        return x + typename S::Tangent(std::cos( newOrientation ) * u.v(),
                                       std::sin( newOrientation ) * u.v(),
                                       u.dtheta());
    }
    
protected:
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-linear state transition function \f$f(x,u)\f$ around the
     * current state \f$x\f$.
     *
     * @note This is only needed when implementing a LinearizedSystemModel,
     *       for usage with an ExtendedKalmanFilter or SquareRootExtendedKalmanFilter.
     *       When using a fully non-linear filter such as the UnscentedKalmanFilter
     *       or its square-root form then this is not needed.
     *
     * @param x The current system state around which to linearize
     * @param u The current system control input
     */
    void updateJacobians( const S& x, const C& u )
    {
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        this->F.setZero();
        
        // partial derivative of x.x() w.r.t. x.x()
        this->F( 0, 0 ) = 1;
        // partial derivative of x.x() w.r.t. x.theta()
        this->F( 0, 2 ) = -std::sin( x.theta() + u.dtheta() ) * u.v();
        
        // partial derivative of x.y() w.r.t. x.y()
        this->F( 1, 1 ) = 1;
        // partial derivative of x.y() w.r.t. x.theta()
        this->F( 1, 2 ) = std::cos( x.theta() + u.dtheta() ) * u.v();
        
        // partial derivative of x.theta() w.r.t. x.theta()
        this->F( 2, 2 ) = 1;
        
        // W = df/dw (Jacobian of state transition w.r.t. the noise)
        this->W.setIdentity();
        // TODO: more sophisticated noise modelling
        //       i.e. The noise affects the the direction in which we move as 
        //       well as the velocity (i.e. the distance we move)
    }
};

} // namespace Robot
} // namespace KalmanExamples

#endif
