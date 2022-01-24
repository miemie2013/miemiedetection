#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <vector>

namespace mmdet {

class Tensor
{
	int dim0;
	int dim1;
	int dim2;
	int dim3;
	int dim4;
	int dim5;
	int dim6;
public:
	Tensor(int dim0=-1, int dim1=-1, int dim2=-1, int dim3=-1, int dim4=-1, int dim5=-1, int dim6=-1)
	{
		this->dim0 = dim0;
		this->dim1 = dim1;
		this->dim2 = dim2;
		this->dim3 = dim3;
		this->dim4 = dim4;
		this->dim5 = dim5;
		this->dim6 = dim6;
	}


	void reshape() {
		;
	}
	int getDim0() {
		return this->dim0;
	}
};

} // namespace mmdet

#endif // __TENSOR_H_

