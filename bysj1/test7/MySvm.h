#ifndef MYSVM_H
#define MYSVM_H
#endif // MYSVM_H
class MySvm : public CvSVM
{
public:
    //获得SVM的决策函数中的alpha数组
   double * get_alpha()
   {
      return this->decision_func->alpha;
   }

   //获得SVM的决策函数中的rho参数,即偏移量
   float get_rho()
   {
      return this->decision_func->rho;
   }
};
