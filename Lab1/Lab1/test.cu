#include <gtest/gtest.h>
#include <iostream>

TEST(SumTest, OnePlusOneEqualsTwo) {
  EXPECT_EQ(sumH(1,1), 2);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}