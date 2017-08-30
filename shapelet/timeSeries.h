
class timeSeries{

      public:
              double *x;
              int length;
              int c;
              double * sumX;
              double * sumX2;
              int originalClass;
			  int originalLength;
			  double * original_x;
			  int samplingRate;

              timeSeries(int n , double * y , int c , int samplingRate);
              ~timeSeries();
              
 
			  int downSample(int s);
              double mean(int i , int len);
              double stdv(int i, int len);
              void computeX();

              int * strt;
              int * lens;
              int d;
              int checkPos(int s , int len);
              void clearPos();
              int insertPos(int i , int len);
			  void init();
 };
 
