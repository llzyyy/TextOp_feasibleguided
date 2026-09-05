#pragma once

#include <vector>
#include <array>
#include <cstdint>

struct KeyMap {
    static constexpr int start = 0;
    static constexpr int select = 1;
    static constexpr int A = 2;
    static constexpr int B = 3;
    static constexpr int X = 4;
    static constexpr int Y = 5;
    static constexpr int up = 6;
    static constexpr int down = 7;
    static constexpr int left = 8;
    static constexpr int right = 9;
    static constexpr int L1 = 10;
    static constexpr int L2 = 11;
    static constexpr int R1 = 12;
    static constexpr int R2 = 13;
    static constexpr int left_stick = 14;
    static constexpr int right_stick = 15;
};

class RemoteController {
public:
    RemoteController();
    
    void set(const std::array<uint8_t, 40>& wireless_remote);
    
    std::array<int, 16> button;
    float lx, ly, rx, ry;
    
private:
    void update_gamepad(const std::array<uint8_t, 40>& data);
};
