#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {
	float* a[1], b[1];
	a[0] = 1.f;
	b[0] = 1.f;
	float* c[1];
	sum<<<1, 1>>>(a, b, c);
	EXPECT_EQ(c[0], 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}