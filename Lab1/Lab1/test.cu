#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {
	const float* a = 1.0, b = 1.0, c = 0.0;
	sum(a,b,c);
	EXPECT_EQ(c, 2.0);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}