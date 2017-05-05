#pragma once

#include<string>
#include <opencv2/core.hpp>

namespace hai {
    struct Annotation {
    public:
        Annotation() noexcept : rectangle{cv::Rect(0, 0, 0, 0)}, description{"empty"}, type{"empty"} {}

        Annotation(std::vector<cv::Point> aContour, std::string aDesc, std::string aType) noexcept : contour{aContour},
                                                                                                     description{aDesc},
                                                                                                     type{aType} {}

        Annotation(const cv::Rect& rect, const std::string& desc, const std::string& aType) : rectangle{rect},
                                                                                              description{desc},
                                                                                              type{aType} {}

        Annotation(cv::Rect&& rect, std::string&& desc, std::string&& aType) : rectangle{std::move(rect)},
                                                                               description{std::move(desc)},
                                                                               type{std::move(aType)} {}

        inline const std::string getDescription() noexcept { return description; }

        inline const std::string getType() noexcept { return type; }

        inline const cv::Rect getRectangle() noexcept { return rectangle; }

        inline const std::vector<cv::Point> getContour() noexcept { return contour; }

    private:
        cv::Rect rectangle;
        std::vector<cv::Point> contour;
        std::string description;
        std::string type;
    };
}

