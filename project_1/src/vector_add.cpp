#include <vector>
#include <algorithm>

using std::vector;

vector<float> cpu_addition(const vector<float> &a, const vector<float> &b) {
    vector<float> results(a);
    std::transform(a.begin(), a.end(), b.cbegin(), b.cend(), std::plus<float>());
    return a;
}

bool check_equal(const vector<float> &a, const vector<float> &b) {
    return std::equal(a.cbegin(), a.cend(), b.cbegin());
}



