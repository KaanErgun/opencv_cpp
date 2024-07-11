#ifndef MY_HELPER_H
#define MY_HELPER_H

#include <string>
#include <sstream>
#include <vector>
#include <iterator>



namespace Helper
{
    void split(const std::string &s, char delim, std::vector<std::string> &result);
    std::vector<std::string> split(const std::string &s, char delim);

    std::string base64_encode(unsigned char const* , unsigned int len);
    std::string base64_decode(std::string const& s);
}

#endif