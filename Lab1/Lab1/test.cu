#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {
	float* c;
	sum(1.f, 1.f, c);
	EXPECT_EQ(c, 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}