#include "timeSeries.h"
#include "xyMatrix.h"
#include <stdio.h>

xyMatrix::xyMatrix(int n, int m)
{
    this->n = n;
    this->m = m;
               
    d = (double **)malloc(sizeof(double *)*n);
	if( d == NULL )
	{
		printf( "ERROR!!! Memory Can't be allocated\n" ); 
		exit(1);
	}
    for( int i = 0 ; i < n ; i++ )
	{
        d[i] = (double *)malloc(sizeof(double)*m);
		if( d[i] == NULL )
		{
			printf(  "ERROR!!! Memory Can't be allocated insinde\n" );
			exit(1);
		}

	}
};




xyMatrix::~xyMatrix()
{
    for( int i = 0 ; i < n ; i++ )
         free(d[i]);
    free(d);             
};

void xyMatrix::computeXY(double *x , int n  , double *y , int m)
{
	this->n = n;
	this->m = m;
    int L = (n>m)?n:m;
    int i , j , k; 
    for( k = 0 ; k < L ; k++ )
    {
		    if( k < n )
			{
				d[k][0] = x[k]*y[0];
				for( i = k+1 , j = 1 ; i < n && j < m ; i++, j++ )
					d[i][j] = d[i-1][j-1]+x[i]*y[j];
			}

			if( k < m )
			{
	            d[0][k] = x[0]*y[k];
		        for( i = 1 , j = k+1 ; i < n && j < m ; i++, j++ )
			       d[i][j] = d[i-1][j-1]+x[i]*y[j];
			}
    }     
};



double xyMatrix::sumXY(int i , int j , int len)
{
	if( i > 0 && j > 0 )
		return d[i+len-1][j+len-1]-d[i-1][j-1];
	else
		return d[i+len-1][j+len-1];

};
