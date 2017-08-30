#include <stdlib.h>
#include <math.h>

class xyMatrix{

      public:
              double ** d;
              int n;
              int m;


			 xyMatrix(int n, int m);
  			  ~xyMatrix();
              

              double sumXY(int i , int j , int len);
              void computeXY(double *x , int n , double *y , int m);
 };
