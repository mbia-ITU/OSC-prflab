/********************************************************
 * Kernels to be optimized for the OS&C prflab.
 * Acknowledgment: This lab is an extended version of the
 * CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "smooth.h" // helper functions for naive_smooth
#include "blend.h"  // helper functions for naive_blend
#include <x86intrin.h> // for SIMD

/* 
 * Please fill in the following struct
 */
student_t student = {
    "mbia",             /* ITU alias */
    "Mikkel Bistrup Andersen",    /* Full name */
    "mbia@itu.dk", /* Email address */
};

/******************************************************************************
 * ROTATE KERNEL
 *****************************************************************************/

// Your different versions of the rotate kernel go here

/* 
 * naive_rotate - The naive baseline version of rotate 
 */
/* stride pattern, visualization (we recommend that you draw this for your functions):
    dst         src
    3 7 B F     0 1 2 3
    2 6 A E     4 5 6 7
    1 5 9 D     8 9 A B
    0 4 8 C     C D E F
 */
char naive_rotate_descr[] = "naive_rotate: Naive baseline implementation";
void naive_rotate(int dim, pixel *src, pixel *dst) 
{
    int i, j;

    for (i = 0; i < dim; i++)
	for (j = 0; j < dim; j++)
	    dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

char row_rotate_descr[] = "row_rotate: rotates as row major";
void row_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;
    for (i = 0; i < dim; i++)
    for (j = 0; j < dim; j++)
        dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

char inner_rotate_descr[] = "inner_rotate: get computations from inner loop";
void inner_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dimj;
    for(j = 0; j < dim; j++) {
        dimj = (dim-1-j) * dim;
        for(i = 0; i < dim; i++) {
            dst[dimj+i] = src[RIDX(i, j, dim)];
        }
    }
}

char unroll_loops_descr[] = "unroll_loops: unroll loops";
void unroll_loops(int dim, pixel *src, pixel *dst)
{
    int i, j, dimi, dimj;

    for (j=0;j<dim;j++)
    {
        dimj = (dim-1-j) * dim;
        for (i=0;i<dim;i+=32)
        {
            dimi = dimj + i;
            dst[dimi+0] = src[RIDX(i+0, j, dim)];
            dst[dimi+1] = src[RIDX(i+1, j, dim)];
            dst[dimi+2] = src[RIDX(i+2, j, dim)];
            dst[dimi+3] = src[RIDX(i+3, j, dim)];
            dst[dimi+4] = src[RIDX(i+4, j, dim)];
            dst[dimi+5] = src[RIDX(i+5, j, dim)];
            dst[dimi+6] = src[RIDX(i+6, j, dim)];
            dst[dimi+7] = src[RIDX(i+7, j, dim)];
            dst[dimi+8] = src[RIDX(i+8, j, dim)];
            dst[dimi+9] = src[RIDX(i+9, j, dim)];
            dst[dimi+10] = src[RIDX(i+10, j, dim)];
            dst[dimi+11] = src[RIDX(i+11, j, dim)];
            dst[dimi+12] = src[RIDX(i+12, j, dim)];
            dst[dimi+13] = src[RIDX(i+13, j, dim)];
            dst[dimi+14] = src[RIDX(i+14, j, dim)];
            dst[dimi+15] = src[RIDX(i+15, j, dim)];
            dst[dimi+16] = src[RIDX(i+16, j, dim)];
            dst[dimi+17] = src[RIDX(i+17, j, dim)];
            dst[dimi+18] = src[RIDX(i+18, j, dim)];
            dst[dimi+19] = src[RIDX(i+19, j, dim)];
            dst[dimi+20] = src[RIDX(i+20, j, dim)];
            dst[dimi+21] = src[RIDX(i+21, j, dim)];
            dst[dimi+22] = src[RIDX(i+22, j, dim)];
            dst[dimi+23] = src[RIDX(i+23, j, dim)];
            dst[dimi+24] = src[RIDX(i+24, j, dim)];
            dst[dimi+25] = src[RIDX(i+25, j, dim)];
            dst[dimi+26] = src[RIDX(i+26, j, dim)];
            dst[dimi+27] = src[RIDX(i+27, j, dim)];
            dst[dimi+28] = src[RIDX(i+28, j, dim)];
            dst[dimi+29] = src[RIDX(i+29, j, dim)];
            dst[dimi+30] = src[RIDX(i+30, j, dim)];
            dst[dimi+31] = src[RIDX(i+31, j, dim)]; 
        }
    }
}

/*not sure if this is allowed?? But by rotating pointers we avoid arrays*/
char helper_pointer_rotate_descr[] = "helper_pointer_rotate: use helper pointer";
void helper_pointer_rotate(size_t from, size_t to, size_t dim, pixel *src, pixel *dst)
{
    size_t i, j, dimj;
    pixel *dsti, *srcj;

    for(j=from;j<to;j++) 
    {
        dimj = (dim-1-j) * dim;
         for(i=0;i<dim;i++)
         {
            dsti = dst + dimj + i;
            srcj = src + j;

            *(dsti + 0) = *(srcj + (i+0)*dim);
            *(dsti + 1) = *(srcj + (i+1)*dim);
            *(dsti + 2) = *(srcj + (i+2)*dim);
            *(dsti + 3) = *(srcj + (i+3)*dim);
            *(dsti + 4) = *(srcj + (i+4)*dim);
            *(dsti + 5) = *(srcj + (i+5)*dim);
            *(dsti + 6) = *(srcj + (i+6)*dim);
            *(dsti + 7) = *(srcj + (i+7)*dim);
            *(dsti + 8) = *(srcj + (i+8)*dim);
            *(dsti + 9) = *(srcj + (i+9)*dim);
            *(dsti + 10) = *(srcj + (i+10)*dim);
            *(dsti + 11) = *(srcj + (i+11)*dim);
            *(dsti + 12) = *(srcj + (i+12)*dim);
            *(dsti + 13) = *(srcj + (i+13)*dim);
            *(dsti + 14) = *(srcj + (i+14)*dim);
            *(dsti + 15) = *(srcj + (i+15)*dim);
            *(dsti + 16) = *(srcj + (i+16)*dim);
            *(dsti + 17) = *(srcj + (i+17)*dim);
            *(dsti + 18) = *(srcj + (i+18)*dim);
            *(dsti + 19) = *(srcj + (i+19)*dim);
            *(dsti + 20) = *(srcj + (i+20)*dim);
            *(dsti + 21) = *(srcj + (i+21)*dim);
            *(dsti + 22) = *(srcj + (i+22)*dim);
            *(dsti + 23) = *(srcj + (i+23)*dim);
            *(dsti + 24) = *(srcj + (i+24)*dim);
            *(dsti + 25) = *(srcj + (i+25)*dim);
            *(dsti + 26) = *(srcj + (i+26)*dim);
            *(dsti + 27) = *(srcj + (i+27)*dim);
            *(dsti + 28) = *(srcj + (i+28)*dim);
            *(dsti + 29) = *(srcj + (i+29)*dim);
            *(dsti + 30) = *(srcj + (i+30)*dim);
            *(dsti + 31) = *(srcj + (i+31)*dim);
         }
    }
}

char pointer_rotate_descr[] = "pointer_rotate: rotates with pointers";
void pointer_rotate(int dim, pixel *src, pixel *dst)
{
    helper_pointer_rotate(0, dim, dim, src, dst);
}

void helper_SIMD_rotate(size_t from, size_t to, size_t dim, pixel *src, pixel *dst)
{
    size_t i, j;

    pixel *dsti, *dstidimj, *srcj;
    register __m256i inc asm("ymm0");
    register __m256i dimv asm("ymm1");
    register __m256i set1i asm("ymm2");
    register __m256i srcv0 asm("ymm3");
    register __m256i srcv1 asm("ymm4");
    register __m256i srcv2 asm("ymm5");
    register __m256i srcv3 asm("ymm6");

    inc = _mm256_set_epi64x(3, 2, 1, 0);
    dimv = _mm256_set1_epi64x(dim);

    for(j=from;j<to;j++)
    {
        dstidimj = dst + (dim-1-j) * dim;
        srcj = src + j;

        for(i=0;i<dim;i+=4)
        {
            dsti = dstidimj + i;
            set1i = _mm256_set1_epi64x(i);
            srcv0 = inc;
            srcv1 = _mm256_add_epi64(srcv0, set1i);
            srcv2 = _mm256_mul_epi32(srcv1, dimv);
            srcv3 = _mm256_i64gather_epi64(srcj, srcv2, sizeof(pixel));
            _mm256_store_si256((__m256i *)dsti, srcv3);
        }
    }
}

char SIMD_rotate_descr[] = "SIMD_rotate: use SIMD";
void SIMD_rotate(int dim, pixel *src, pixel *dst)
{
    helper_SIMD_rotate(0, dim, dim, src, dst);
}

char rotate_descr[] = "rotate: Current working version";
void rotate(int dim, pixel *src, pixel *dst)
{
    pointer_rotate(dim, src, dst);
}

/*
 * register_rotate_functions - Register all of your different versions
 *     of the rotate kernel with the driver by calling the
 *     add_rotate_function() for each test function.
 */
void register_rotate_functions() 
{
    add_rotate_function(&rotate, rotate_descr);
    add_rotate_function(&naive_rotate, naive_rotate_descr);
    add_rotate_function(&row_major_rotate, row_major_rotate_descr);
    add_rotate_function(&hoist_rotate, hoist_rotate_descr);
    add_rotate_function(&unroll_rotate, unroll_rotate_descr);
    add_rotate_function(&pointers_rotate, pointers_rotate_descr);
    add_rotate_function(&simd_rotate, simd_rotate_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * ROTATE_T KERNEL
 *****************************************************************************/



/* 
 * rotate_t - Your current working version of rotate_t
 * IMPORTANT: This is the version you will be graded on
 */
char rotate_t_descr[] = "rotate_t: Current working version";
void rotate_t(int dim, pixel *src, pixel *dst)
{
    naive_rotate(dim, src, dst);
}

/*********************************************************************
 * register_rotate_t_functions - Register all of your different versions
 *     of the rotate_t kernel with the driver by calling the
 *     add_rotate_t_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 *********************************************************************/

void register_rotate_t_functions() 
{
    add_rotate_t_function(&rotate_t, rotate_t_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND KERNEL
 *****************************************************************************/

// Your different versions of the blend kernel go here.

char naive_blend_descr[] = "naive_blend: Naive baseline implementation";
void naive_blend(int dim, pixel *src, pixel *dst) // reads global variable `pixel bgc`
{
    int i, j;

    for (i = 0; i < dim; i++)
	for (j = 0; j < dim; j++)
	    blend_pixel(&src[RIDX(i, j, dim)], &dst[RIDX(i, j, dim)], &bgc); // `blend_pixel` defined in blend.c
}

char blend_descr[] = "blend: Current working version";
void blend(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

/*
 * register_blend_functions - Register all of your different versions
 *     of the blend kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_functions() {
    add_blend_function(&blend, blend_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND_V KERNEL
 *****************************************************************************/

// Your different versions of the blend_v kernel go here
// (i.e. with vectorization, aka. SIMD).

char blend_v_descr[] = "blend_v: Current working version";
void blend_v(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

/*
 * register_blend_v_functions - Register all of your different versions
 *     of the blend_v kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_v_functions() {
    add_blend_v_function(&blend_v, blend_v_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * SMOOTH KERNEL
 *****************************************************************************/

// Your different versions of the smooth kernel go here

/*
 * naive_smooth - The naive baseline version of smooth 
 */
char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
void naive_smooth(int dim, pixel *src, pixel *dst) 
{
    int i, j;

    for (i = 0; i < dim; i++)
	for (j = 0; j < dim; j++)
	    dst[RIDX(i, j, dim)] = avg(dim, i, j, src); // `avg` defined in smooth.c
}

char smooth_descr[] = "smooth: Current working version";
void smooth(int dim, pixel *src, pixel *dst)
{
  naive_smooth(dim, src, dst);
}

/*
 * register_smooth_functions - Register all of your different versions
 *     of the smooth kernel with the driver by calling the
 *     add_smooth_function() for each test function.
 */

void register_smooth_functions() {
    add_smooth_function(&smooth, smooth_descr);
    /* ... Register additional test functions here */
}
