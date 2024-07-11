#include "webapi.h"

#include <curl/curl.h>
#include <cstring>
#include <memory>
#include <iostream>

bool WebAPI::getAuthToken(std::string &auth_token)
{
    int httpCode = 0;
    std::unique_ptr<std::string> httpData(new std::string());

    curl_global_init(CURL_GLOBAL_ALL);
    CURL *curl = curl_easy_init();

    curl_easy_setopt(curl, CURLOPT_POST, 1);
    curl_easy_setopt(curl, CURLOPT_URL, "https://spapi.residents.net.au/api/auth/token"); //web API adresi
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WebAPI::curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, httpData.get());

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "cache-control: no-cache");
    headers = curl_slist_append(headers, "content-type: application/x-www-form-urlencoded");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "grant_type=password&username=sa%40sp.sp&password=test123&role="); //web apinin kullanici adi ve sifresi

    CURLcode ret = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    if (httpCode != 200)
    {
        std::cout << "Response code;" << std::endl;
        std::cout << "HTTP data was:\n"
                  << *httpData.get() << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    if (ret != CURLE_OK)
    {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(ret));
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    std::string temp = *httpData.get();

    int pos = temp.find(":");
    std::string token = temp.substr(pos + 2);

    std::string::size_type pos1 = token.find(',');
    if (pos1 != std::string::npos)
    {
        token = token.substr(0, pos1 - 1);
    }

    auth_token = std::move(token);

    return true;
}

std::size_t WebAPI::curl_write_callback(const char *in, std::size_t size, std::size_t num, std::string *out)
{
    const std::size_t totalBytes(size * num);
    out->append(in, totalBytes);
    return totalBytes;
}

bool WebAPI::uploadPlateLog(std::string auth_token, std::string json_str)
{

    int httpCode = 0;

    curl_global_init(CURL_GLOBAL_ALL);
    CURL *curl = curl_easy_init();
    CURLcode curlRes;
    std::string authorization = "Authorization: Bearer " + auth_token;

    curl_easy_setopt(curl, CURLOPT_URL, "https://spapi.residents.net.au/api/v1.0/carmanager/save-car-access-log");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_POST, 1);
    // curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, authorization.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());

    // std::cout << "JSON Veri: " << json_veri << std::endl;

    std::unique_ptr<std::string> httpData(new std::string());

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WebAPI::curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, httpData.get());

    curlRes = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    // std::cout << "CURL CODE: " << (int)res << std::endl;

    if (httpCode != 200)
    {
        std::cout << "Response code;" << std::endl;
        std::cout << "HTTP data was:\n"
                  << *httpData.get() << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    if (curlRes != CURLE_OK)
    {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(curlRes));
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    Json::Value jsonData;
    Json::Reader jsonReader;

    if (jsonReader.parse(*httpData.get(), jsonData))
    {
        std::cout << "Successfully parsed JSON data" << std::endl;
        std::cout << "\nJSON data received:" << std::endl;
        std::cout << jsonData.toStyledString() << std::endl;

        std::cout << "Success: " << jsonData["success"].asString() << std::endl;
    }
    else
    {
        std::cout << "Could not parse HTTP data as JSON" << std::endl;
        std::cout << "HTTP data was:\n"
                  << *httpData.get() << std::endl;
        return 1;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl = NULL;

    return true;
}