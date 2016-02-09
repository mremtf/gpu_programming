#ifndef _TIMER_HPP__
#define _TIMER_HPP__

#include <chrono>

using namespace std::chrono;
using hrc = high_resolution_clock;

class simple_timer {
  private:
    hrc::time_point a, b;

  public:
    void begin() { a = hrc::now(); }
    void end() { b = hrc::now(); }
    template <typename time_unit>
    double elapsed() {
        return duration<double, time_unit>(b - a).count();
    }
    double ms_elapsed() {
    	return elapsed<std::milli>();
    }
};

#endif