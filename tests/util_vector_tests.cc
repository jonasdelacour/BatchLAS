#include <gtest/gtest.h>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>

TEST(SyclVectorTest, DefaultConstruction) {
    SyclVector<int> vec;
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_EQ(vec.data(), nullptr);
}

TEST(SyclVectorTest, SizeConstruction) {
    SyclVector<int> vec(5);
    EXPECT_EQ(vec.size(), 5);
    EXPECT_GE(vec.capacity(), 5);
    EXPECT_NE(vec.data(), nullptr);
}

TEST(SyclVectorTest, SizeValueConstruction) {
    SyclVector<int> vec(5, 42);
    EXPECT_EQ(vec.size(), 5);
    EXPECT_GE(vec.capacity(), 5);
    
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], 42);
    }
}

TEST(SyclVectorTest, CopyConstruction) {
    SyclVector<int> original(5, 42);
    SyclVector<int> copy(original);
    
    EXPECT_EQ(copy.size(), 5);
    for (size_t i = 0; i < copy.size(); i++) {
        EXPECT_EQ(copy[i], 42);
    }
    
    // Ensure they are independent copies
    original[0] = 10;
    EXPECT_EQ(original[0], 10);
    EXPECT_EQ(copy[0], 42);
}

TEST(SyclVectorTest, MoveConstruction) {
    SyclVector<int> original(5, 42);
    int* original_data = original.data();
    
    SyclVector<int> moved(std::move(original));
    
    EXPECT_EQ(moved.size(), 5);
    EXPECT_EQ(moved.data(), original_data);
    
    EXPECT_EQ(original.size(), 0);
    EXPECT_EQ(original.capacity(), 0);
    EXPECT_EQ(original.data(), nullptr);
}

TEST(SyclVectorTest, CopyAssignment) {
    SyclVector<int> original(5, 42);
    SyclVector<int> copy;
    
    copy = original;
    
    EXPECT_EQ(copy.size(), 5);
    for (size_t i = 0; i < copy.size(); i++) {
        EXPECT_EQ(copy[i], 42);
    }
    
    // Ensure they are independent copies
    original[0] = 10;
    EXPECT_EQ(original[0], 10);
    EXPECT_EQ(copy[0], 42);
}

TEST(SyclVectorTest, MoveAssignment) {
    SyclVector<int> original(5, 42);
    int* original_data = original.data();
    
    SyclVector<int> moved;
    moved = std::move(original);
    
    EXPECT_EQ(moved.size(), 5);
    EXPECT_EQ(moved.data(), original_data);
    
    EXPECT_EQ(original.size(), 0);
    EXPECT_EQ(original.capacity(), 0);
    EXPECT_EQ(original.data(), nullptr);
}

TEST(SyclVectorTest, Resize) {
    SyclVector<int> vec(3, 1);
    EXPECT_EQ(vec.size(), 3);
    
    // Resize larger
    vec.resize(5);
    EXPECT_EQ(vec.size(), 5);
    EXPECT_GE(vec.capacity(), 5);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(vec[i], 1);
    }
    
    // Resize smaller
    vec.resize(2);
    EXPECT_EQ(vec.size(), 2);
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], 1);
    }
}

TEST(SyclVectorTest, ResizeWithValue) {
    SyclVector<int> vec(3, 1);
    
    // Resize with new value
    vec.resize(5, 42);
    EXPECT_EQ(vec.size(), 5);
    
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(vec[i], 1); // Original values preserved
    }
    
    for (size_t i = 3; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], 42); // New elements initialized with value
    }
}

TEST(SyclVectorTest, Reserve) {
    SyclVector<int> vec;
    vec.reserve(10);
    
    EXPECT_EQ(vec.size(), 0);
    EXPECT_GE(vec.capacity(), 10);
    
    // Add elements
    for (int i = 0; i < 5; i++) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.size(), 5);
    EXPECT_GE(vec.capacity(), 10);
}

TEST(SyclVectorTest, PushBack) {
    SyclVector<int> vec;
    
    vec.push_back(1);
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec[0], 1);
    
    vec.push_back(2);
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    
    // Test capacity growth
    for (int i = 3; i <= 10; i++) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.size(), 10);
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(vec[i], i+1);
    }
}

TEST(SyclVectorTest, PopBack) {
    SyclVector<int> vec;
    for (int i = 1; i <= 5; i++) {
        vec.push_back(i);
    }
    
    int popped = vec.pop_back();
    EXPECT_EQ(popped, 5);
    EXPECT_EQ(vec.size(), 4);
    
    popped = vec.pop_back();
    EXPECT_EQ(popped, 4);
    EXPECT_EQ(vec.size(), 3);
    
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
}

TEST(SyclVectorTest, Clear) {
    SyclVector<int> vec(5, 42);
    size_t original_capacity = vec.capacity();
    
    vec.clear();
    
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.capacity(), original_capacity); // Capacity unchanged
}

TEST(SyclVectorTest, IndexOperator) {
    SyclVector<int> vec(5);
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = i * 10;
    }
    
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], i * 10);
    }
}

TEST(SyclVectorTest, AtMethod) {
    SyclVector<int> vec(5);
    for (size_t i = 0; i < vec.size(); i++) {
        vec.at(i) = i * 10;
    }
    
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec.at(i), i * 10);
    }
}

TEST(SyclVectorTest, FrontBack) {
    SyclVector<int> vec;
    for (int i = 1; i <= 5; i++) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.front(), 1);
    EXPECT_EQ(vec.back(), 5);
    
    vec.front() = 10;
    vec.back() = 50;
    
    EXPECT_EQ(vec[0], 10);
    EXPECT_EQ(vec[4], 50);
}

TEST(SyclVectorTest, Swap) {
    SyclVector<int> vec1(3, 10);
    SyclVector<int> vec2(5, 20);
    
    vec1.swap(vec2);
    
    EXPECT_EQ(vec1.size(), 5);
    EXPECT_EQ(vec2.size(), 3);
    
    for (size_t i = 0; i < vec1.size(); i++) {
        EXPECT_EQ(vec1[i], 20);
    }
    
    for (size_t i = 0; i < vec2.size(); i++) {
        EXPECT_EQ(vec2[i], 10);
    }
    
    // Test global swap
    swap(vec1, vec2);
    
    EXPECT_EQ(vec1.size(), 3);
    EXPECT_EQ(vec2.size(), 5);
}

TEST(SyclVectorTest, ToSpan) {
    SyclVector<int> vec(5, 42);
    Span<int> span = vec.to_span();
    
    EXPECT_EQ(span.size(), 5);
    EXPECT_EQ(span.data(), vec.data());
    
    for (size_t i = 0; i < span.size(); i++) {
        EXPECT_EQ(span[i], 42);
    }
}

TEST(SyclVectorTest, Subspan) {
    SyclVector<int> vec;
    for (int i = 0; i < 10; i++) {
        vec.push_back(i);
    }
    
    // Full subspan
    Span<int> sub1 = vec.subspan(0, vec.size());
    EXPECT_EQ(sub1.size(), 10);
    
    // Partial from start
    Span<int> sub2 = vec.subspan(0, 5);
    EXPECT_EQ(sub2.size(), 5);
    for (size_t i = 0; i < sub2.size(); i++) {
        EXPECT_EQ(sub2[i], i);
    }
    
    // Partial from middle
    Span<int> sub3 = vec.subspan(5, 5);
    EXPECT_EQ(sub3.size(), 5);
    for (size_t i = 0; i < sub3.size(); i++) {
        EXPECT_EQ(sub3[i], i + 5);
    }
    
    // From offset to end
    Span<int> sub4 = vec.subspan(7);
    EXPECT_EQ(sub4.size(), 3);
    for (size_t i = 0; i < sub4.size(); i++) {
        EXPECT_EQ(sub4[i], i + 7);
    }
}

TEST(SyclVectorTest, Fill) {
    SyclVector<int> vec(10);
    vec.fill(42);
    
    for (size_t i = 0; i < vec.size(); i++) {
        EXPECT_EQ(vec[i], 42);
    }
}

TEST(SyclVectorTest, Equality) {
    SyclVector<int> vec1(5, 10);
    SyclVector<int> vec2(5, 10);
    SyclVector<int> vec3(5, 20);
    SyclVector<int> vec4(3, 10);
    
    EXPECT_TRUE(vec1 == vec2);
    EXPECT_FALSE(vec1 == vec3);
    EXPECT_FALSE(vec1 == vec4);
}