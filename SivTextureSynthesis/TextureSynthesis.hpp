#pragma once
#include "Common.hpp"

class TextureSynthesis {
public:
    explicit TextureSynthesis(Image input, Size size, int32 kernelSize = 11) noexcept;
    explicit TextureSynthesis(Image input, int32 width, int32 height, int32 kernelSize = 11) noexcept;

    Image synthesize() noexcept;
    void synthesizeAsync() noexcept;
    void fill(DynamicTexture& dtexture) noexcept;
    void exit() noexcept;

private:
    void init() noexcept;
    Grid<double> gaussianMask(const int32 width, const int32 height, const double sigma) const noexcept;
    Array<Point> neighboringPixelIndices() noexcept;
    Grid<double> normalizedSSD(const Point c) noexcept;
    Array<Point> candidateIndices(const Grid<double>& ssd) noexcept;

private:
    static constexpr int32 DX[8] = { -1, 0, 1, 0,-1, 1, 1,-1 };
    static constexpr int32 DY[8] = { 0, 1, 0,-1, 1, 1,-1,-1 };
    static constexpr double ErrorThreshold = 0.1;
    const Image m_sample;
    const int32 m_kernelSize;
    Grid<double> m_gaussianMask;
    Image m_generatedImage;

    Grid<bool> m_mask;
    int32 m_remainingPixels;
    AsyncTask<Image> m_asyncTask;
    bool m_exit = false;
};
