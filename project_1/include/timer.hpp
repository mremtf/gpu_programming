#include <chrono>

using std::chrono::high_resolution_clock;  // steady_clock?

class simple_timer {
  private:
    time_point a, b;

  public:
    void begin() { a = now(); }
    void end() { b = now(); }
    template <typename time_unit>
    double elapsed() {
        return duration_cast<double, time_unit>(b - a);
    }
    double ms_elapsed() {
    	return elapsed<std::chrono::milliseconds>();
    }
};