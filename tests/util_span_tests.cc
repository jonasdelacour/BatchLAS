#include <gtest/gtest.h>
#include <util/sycl-span.hh>
#include <vector>
#include <array>

TEST(SyclSpanTest, DefaultConstruction) {
    Span<int> span;
    EXPECT_EQ(span.size(), 0);
    EXPECT_TRUE(span.empty());
    EXPECT_EQ(span.data(), nullptr);
}

TEST(SyclSpanTest, ConstructFromPointerAndSize) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    EXPECT_EQ(span.size(), 5);
    EXPECT_FALSE(span.empty());
    EXPECT_EQ(span.data(), data);
    EXPECT_EQ(span[0], 1);
    EXPECT_EQ(span[4], 5);
}

TEST(SyclSpanTest, ConstructFromBeginEnd) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, data + 5);
    EXPECT_EQ(span.size(), 5);
    EXPECT_FALSE(span.empty());
    EXPECT_EQ(span.data(), data);
    EXPECT_EQ(span[0], 1);
    EXPECT_EQ(span[4], 5);
}

TEST(SyclSpanTest, ConstructFromSingleValue) {
    int value = 42;
    Span<int> span(value);
    EXPECT_EQ(span.size(), 1);
    EXPECT_FALSE(span.empty());
    EXPECT_EQ(span[0], 42);
}

TEST(SyclSpanTest, CopyConstruction) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> original(data, 5);
    Span<int> copy(original);
    
    EXPECT_EQ(copy.size(), 5);
    EXPECT_EQ(copy.data(), data);
    EXPECT_EQ(copy[0], 1);
    EXPECT_EQ(copy[4], 5);
}

TEST(SyclSpanTest, Subspan) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    
    Span<int> sub1 = span.subspan(2);
    EXPECT_EQ(sub1.size(), 3);
    EXPECT_EQ(sub1.data(), data + 2);
    EXPECT_EQ(sub1[0], 3);
    
    Span<int> sub2 = span.subspan(1, 3);
    EXPECT_EQ(sub2.size(), 3);
    EXPECT_EQ(sub2.data(), data + 1);
    EXPECT_EQ(sub2[0], 2);
    EXPECT_EQ(sub2[2], 4);
}

TEST(SyclSpanTest, IteratorAccess) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    
    int sum = 0;
    for (auto it = span.begin(); it != span.end(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 15);
}

TEST(SyclSpanTest, FrontBack) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    
    EXPECT_EQ(span.front(), 1);
    EXPECT_EQ(span.back(), 5);
}

TEST(SyclSpanTest, SizeBytes) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    
    EXPECT_EQ(span.size_bytes(), 5 * sizeof(int));
}

TEST(SyclSpanTest, AsSpan) {
    int data[5] = {1, 2, 3, 4, 5};
    Span<int> span(data, 5);
    
    Span<char> charSpan = span.template as_span<char>();
    EXPECT_EQ(charSpan.size(), 5 * sizeof(int));
}