#ifndef WEBAPI_H
#define WEBAPI_H

#include <cstdint>
#include <string>

#include "../thirdparty/json/json/json.h"

class WebAPI {
public:
    static bool getAuthToken(std::string& auth_token);
    static bool uploadPlateLog(std::string auth_token, std::string json_str);

    static std::size_t curl_write_callback(
        const char *in,
        std::size_t size,
        std::size_t num,
        std::string *out);
};
#endif