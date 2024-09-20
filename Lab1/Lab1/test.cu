#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {
	float a[32], b[32];
	a[0] = 1.f;
	b[0] = 1.f;
	float c[32];
	sum<<<1, 32>>>(a, b, c);
	cudaDeviceSynchronize();
	
	EXPECT_EQ(c[0], 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}