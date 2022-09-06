#include "TextureSynthesis.hpp"

inline constexpr double difference(const Color a, const Color b) noexcept {
    double diff = 0;
    diff += (a.r - b.r) / 255.0 * (a.r - b.r) / 255.0;
    diff += (a.g - b.g) / 255.0 * (a.g - b.g) / 255.0;
    diff += (a.b - b.b) / 255.0 * (a.b - b.b) / 255.0;
    return diff;
}

TextureSynthesis::TextureSynthesis(Image input, Size size, int32 kernelSize) noexcept
    : m_sample(input)
    , m_generatedImage(size, Color(0, 0, 0, 0))
    , m_kernelSize(kernelSize)
    , m_remainingPixels(size.area() - 9)
    , m_mask(size)
    , m_gaussianMask(gaussianMask(kernelSize, kernelSize, kernelSize / 6.4)) {

    init();

}

TextureSynthesis::TextureSynthesis(Image input, int32 width, int32 height, int32 kernelSize) noexcept
    : m_sample(input)
    , m_generatedImage(width, height, Color(0, 0, 0, 0))
    , m_kernelSize(kernelSize)
    , m_remainingPixels(width * height - 9)
    , m_mask(width, height)
    , m_gaussianMask(gaussianMask(kernelSize, kernelSize, kernelSize / 6.4)) {

    init();

}

Image TextureSynthesis::synthesize() noexcept {
    while (m_remainingPixels > 0 && not m_exit) {
        const auto cs = neighboringPixelIndices();
        for (const auto c : cs) {
            auto ssd = normalizedSSD(c);
            auto indices = candidateIndices(ssd);
            Point selected = Sample(indices);
            selected += Point{ m_kernelSize / 2, m_kernelSize / 2 };
            m_generatedImage[c] = m_sample[selected];
            m_mask[c] = true;
            m_remainingPixels--;
        }
    }
    return m_generatedImage;
}

void TextureSynthesis::synthesizeAsync() noexcept {
    m_asyncTask = Async([this]() { return this->synthesize(); });
}

void TextureSynthesis::fill(DynamicTexture& dtexture) noexcept {
    dtexture.fill(m_generatedImage);
}

void TextureSynthesis::exit() noexcept {
    m_exit = true;
}

void TextureSynthesis::init() noexcept {
    Point seed = { Random(m_sample.width() - 2), Random(m_sample.height() - 2) };
    for (const auto d : step(Size{ 3, 3 })) {
        Point point = d + ((Point{ m_generatedImage.size() } / 2) - Point{1, 1});
        m_generatedImage[point] = m_sample[seed + d];
        m_mask[point] = true;
    }
}

Grid<double> TextureSynthesis::gaussianMask(const int32 width, const int32 height, const double sigma) const noexcept {
    const double s = 2.0 * sigma * sigma;
    double sum = 0;
    Grid<double> result(width, height);
    for (const auto y : step(height)) {
        for (const auto x : step(width)) {
            int32 px = x - width / 2;
            int32 py = y - height / 2;
            double r = px * px + py * py;
            result[y][x] = Math::Exp(-r / s) / (Math::Pi * s);
            sum += result[y][x];
        }
    }
    for (auto p : result) {
        p /= sum;
    }
    return result;
}

Array<Point> TextureSynthesis::neighboringPixelIndices() noexcept {
    HashTable<Point, int32> neighborCount;
    const auto [width, height] = m_generatedImage.size();
    for (const auto y : step(height)) {
        for (const auto x : step(width)) {
            if (not m_mask[y][x]) {
                continue;
            }
            for (auto d : step(8)) {
                auto nx = x + DX[d];
                auto ny = y + DY[d];
                if (Rect{ width, height }.contains(Point{ nx, ny })) {
                    if (not m_mask[ny][nx]) {
                        neighborCount[{ nx, ny }]++;
                    }
                }
            }
        }
    }
    Array<Point> result;
    for (const auto [p, _] : neighborCount) {
        result << p;
    }
    // result.shuffle();
    result.sort_by([&](const auto& a, const auto& b) {
        return neighborCount[a] > neighborCount[b];
    });
    return result;
}

Grid<double> TextureSynthesis::normalizedSSD(const Point c) noexcept {
    int32 pad = m_kernelSize / 2;
    Size ssdSize{ m_sample.width() - m_kernelSize + 1, m_sample.height() - m_kernelSize + 1 };
    Grid<double> ssd(ssdSize);
    Grid<double> allMask = m_gaussianMask;

    double totalWeight = 0;
    for (auto b : step({ m_kernelSize, m_kernelSize })) {
        Point d = b + c - Point{ pad, pad };
        if (not Rect{ m_generatedImage.size() }.contains(d)) {
            allMask[b] = 0;
            continue;
        }
        if (not m_mask[d]) {
            allMask[b] = 0;
        }
        totalWeight += allMask[b.y][b.x];
    }
    for (auto a : step({ ssdSize })) {
        for (auto b : step({ m_kernelSize, m_kernelSize })) {
            Point d = b + c - Point{ pad, pad };
            if (not Rect{ m_generatedImage.size() }.contains(d)) {
                continue;
            }
            if (m_generatedImage[d].a == 0) {
                continue;
            }
            double diff = difference(m_generatedImage[d], m_sample[a + b]);
            ssd[a] += diff * allMask[b] / totalWeight;
        }
    }
    return ssd;
}

Array<Point> TextureSynthesis::candidateIndices(const Grid<double>& ssd) noexcept {
    double minSSD = *std::min_element(ssd.begin(), ssd.end());
    double minThreshold = minSSD * (1 + ErrorThreshold);
    Array<Point> result;
    for (auto point : step(ssd.size())) {
        if (ssd[point] <= minThreshold) {
            result << point;
        }
    }
    return result;
}
