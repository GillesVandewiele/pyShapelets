#include <stdlib.h>
#include <math.h>

struct projection{
       
       int tID;
       int pos;
       int length;
       double distance;
       int cls;
       };

class orderLine{
    
    public:  
      int shapeletID;
      int shapeletPos;
      int shapeletLength;
      
      int N;
	  int curN;
      int nCls;
	  double entropy;
	  int * clsHist;


      int * leftClsHist;
      int * rightClsHist;
      
      double leftTotal;
      double rightTotal;
      double leftEntropy;
      double rightEntropy;

	  double informationGain;
	  double gap;
	  double splitDist;
      int splitPos;

	  double tIG;

      struct projection * line;
      
      orderLine();
      //orderLine(orderLine & p);
      orderLine(int n , int nC , int * cH);
	  ~orderLine();
      double findBestSplit();


	  void copy(orderLine * src );

	  void reset(int sId, int sPos, int sLen);
      void insert(int i , projection p);
      double computeInformationGain();
	  double minGap(int j);
	  double gapDist(int j);
	  double shiftEntropy(double shftAmount);
	  double findEntropies();
	  void mergeTwoLines(orderLine * L);
	  double UB(double shiftAmount, int splitPos);
	  double TUB(double shiftAmount);
      
      };
