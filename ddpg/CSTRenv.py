import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import gym
from gym import spaces


class VV:
  """
  Assume isothermal reactor for simplicity
  """
  def __init__(self):
    self.Ca0 =  5.1        # Inlet feed concentration (mol/m^3)
    #self.T =  114.2       # reactor temperature (degC)
    self.T0 = 135

    self.k10 = 1.287e12    # A->B Pre-exponential factor (1/hr)
    self.k20 = 1.287e12    # B->C Pre-exponential factor (1/hr)
    self.k30 = 9.043e9     # 2A->D Pre-exponential factor (1/hr)
    self.E1 = 9758.3       # A->B Activation Energy (K)
    self.E2 = 9758.3       # B->C Activation Energy (K)
    self.E3 = 8560         # 2A->D Activation Energy (K)
    self.deltaH1 = 4.2  #1st reaction enthalpy (kJ/mol)
    self.deltaH2 = -11 #2nd reaction enthalpy (kJ/mol)
    self.deltaH3 = -41.85 #3rd reaction enthalpy (kJ/mol)
    self.rho = 0.9342 #density (kg/L)
    self.cp = 3.01 #heat capacit (kJ/kg K)
    self.Ar = 0.215 #jacket area (m2)
    self.Kw = 4032 #jacket heat transfer coefficient (kJ/h m2 K)
    
    self.VR = 10.0         # Reactor volume (l)
    self.Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    self.A_R = 0.215 # Area of reactor wall [m^2]
    self.m_k = 5.0 # Coolant mass[kg]

    # steady state value of VV
    self.Ca_ss = 2.14
    self.Cb_ss = 1.09
    self.F_ss = 141.9
    self.F_k = 100

  def arrenhius(self, T):
    self.k1 = self.k10*np.exp(-self.E1/(T+273.15))
    self.k2 = self.k20*np.exp(-self.E2/(T+273.15))
    self.k3 = self.k30*np.exp(-self.E3/(T+273.15))

  def linearlized_dynamics(self, s, t):
    
    Ca_dev, Cb_dev, T_dev, Tk_dev, u = s
    self.arrenhius(self.T_ss + T_dev)

    dCadt_lin = (-self.F_ss / self.VR - self.k1 - 2*self.k3*self.Ca_ss) * Ca_dev \
          + (self.Ca0-self.Ca_ss) / self.VR * u
    dCbdt_lin = (-self.F_ss / self.VR - self.k2) * Cb_dev \
          + self.k1*Ca_dev + (-self.Cb_ss / self.VR) * u
    dudt = 0

    return [dCadt_lin, dCbdt_lin, dudt]


class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time
        
        
        
class CSTRenv(gym.Env):
    def __init__(self):
        super(CSTRenv, self).__init__()
        
        num_PID = 2

        # State Space
        #self.state_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([50.0, 50.0, 50.0]), dtype=np.float32)
        self.state_space = spaces.Box(low=np.zeros(2*num_PID), high=50*np.ones(2*num_PID), dtype=np.float32)

        # Action Space
        #self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=-1*np.ones(2*num_PID), high=np.ones(2*num_PID), dtype=np.float32)

        # initial condition
        self.F_init = 140.0
        self.Ca_init = 2.2291     # Concentration of A in CSTR (mol/l)
        self.Cb_init = 1.0417     # Concentration of B in CSTR (mol/l)
        
        # initial state
        self.Kp = 5
        self.Ki = 0
        #self.Kd = 0
    
        self.Ca_setpoint = 3.0    
        self.T_setpoint =   
        self.sample_time = 1/2
        
        self.alpha = 1
        self.beta = 1

        # initialize
        self.reset()

    def step(self, action):
        #assert self.action_space.contains(action)   
        
        # update state
        PID_params = {"Kp" : self.Kp + self.alpha*action[0],
                      "Ki" : self.Ki + self.alpha*action[1]}
                      #"Kd" : self.Kd + self.alpha*action[2]}
        #assert self.state_space.contains(np.array(PID_params.values()))
        
        # VV system
        vv_system = VV()
        
        # PID control
        pid = PID(current_time = 0.0)
        pid.setSampleTime(self.sample_time)
        pid.setKp(PID_params["Kp"])
        pid.setKi(PID_params["Ki"])
        #pid.setKd(PID_params["Kd"])
        pid.SetPoint = self.Ca_setpoint - vv_system.Ca_ss
        
        Ca = [self.Ca_init]
        Cb = [self.Cb_init]
        F = [self.F_init] # 실제 우리가 관측하게 될 유량
        input = [self.F_init-vv_system.F_ss] #PID가 뱉어주는 입력값
        
        current_time = 0.0
        sample_num = 1

        while(sample_num <= 60):
            Ca_dev_init = Ca[-1] - vv_system.Ca_ss
            Cb_dev_init = Cb[-1] - vv_system.Cb_ss

            s_init = (Ca_dev_init, Cb_dev_init, input[-1]) # dynamics에 넣는건 deviation variables
            t = np.linspace(current_time, current_time + self.sample_time, num=int(self.sample_time*60))
            s = odeint(func=vv_system.linearlized_dynamics, y0=s_init, t=t) #sample time 만큼 ode 풀기
            current_Ca_dev, current_Cb_dev = [s[-1][0], s[-1][1]]

            current_time += self.sample_time
            pid.update(current_Cb_dev, current_time) # PID도 deviation variable로
            input.append(pid.output)

            for i in range(int(self.sample_time*60)):
                Ca.append(s.T[0][i]+vv_system.Ca_ss)
                Cb.append(s.T[1][i]+vv_system.Cb_ss)
                F.append(s.T[2][i]+vv_system.F_ss)

            sample_num += 1

        # calculate reward
        t_conv = np.argmin(Ca[-1]==Ca)/sample_num
        reward = -abs(Ca[-1] - self.Ca_setpoint) - self.beta*t_conv
        done = False
        if abs(reward) < 0.02*self.Ca_setpoint:
            done = True  
            
        return np.array(list(PID_params.values())), reward, done, {}

    def reset(self):
        # reset env to initial state
        self.Kp = 5
        self.Ki = 5
        #self.Kd = 0
        
        return np.array([self.Kp, self.Ki, self.Kd])