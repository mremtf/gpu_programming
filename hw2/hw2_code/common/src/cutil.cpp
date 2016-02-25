/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/


/* CUda UTility Library */

/* Credit: Cuda team for the PGM file reader / writer code. */

// includes, file
#include <cutil.h>

// includes, system
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string.h>

// includes, cuda
#include <builtin_types.h>

// includes, common
#include <cmd_arg_reader.h>
#include <error_checker.h>
#include <stopwatch.h>
#include <bank_checker.h>

// includes, system

// namespace unnamed (internal)
namespace 
{  
    // variables

    //! size of PGM file header 
    const unsigned int PGMHeaderSize = 0x40;

    // types

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterFromUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterFromUByte<unsigned int> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned int operator()( const unsigned char& val) {
            return static_cast<unsigned int>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned short
    template<>
    struct ConverterFromUByte<unsigned short> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned short operator()( const unsigned char& val) 
        {
            return static_cast<unsigned short>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned float
    template<>
    struct ConverterFromUByte<float> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()( const unsigned char& val) 
        {
            return static_cast<float>( val) / 255.0f;
        }
    };

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterToUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<unsigned int> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const unsigned int& val) 
        {
            return static_cast<unsigned char>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned short
    template<>
    struct ConverterToUByte<unsigned short> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const unsigned short& val) 
        {
            return static_cast<unsigned char>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned float
    template<>
    struct ConverterToUByte<float> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const float& val) 
        {
            return static_cast<unsigned char>( val * 255.0f);
        }
    };


    // functions

    ////////////////////////////////////////////////////////////////////////////// 
    //! Compare two arrays of arbitrary type       
    //! @return  true if \a reference and \a data are identical, otherwise false
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    //! @param epsilon    epsilon to use for the comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T, class S>
    CUTBoolean  
    compareData( const T* reference, const T* data, const unsigned int len, 
                 const S epsilon) 
    {
        CUT_CONDITION( epsilon >= 0);

        bool result = true;

        for( unsigned int i = 0; i < len; ++i) {

            T diff = reference[i] - data[i];
            bool comp = (diff <= epsilon) && (diff >= -epsilon);
            result &= comp;

#ifdef _DEBUG
            if( ! comp) 
            {
                std::cerr << "ERROR, i = " << i << ",\t " 
                    << reference[i] << " / "
                    << data[i] 
                    << " (reference / data)\n";
            }
#endif
        }

        return (result) ? CUTTrue : CUTFalse;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Read file \filename and return the data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    cutReadFile( const char* filename, T** data, unsigned int* len, bool verbose) 
    {
        // check input arguments
        CUT_CONDITION( NULL != filename);
        CUT_CONDITION( NULL != len);

        // intermediate storage for the data read
        std::vector<T>  data_read;

        // open file for reading
        std::fstream fh( filename, std::fstream::in);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutReadFile() : Opening file failed." << std::endl;
            return CUTFalse;
        }

        // read all data elements 
        T token;
        while( fh.good()) 
        {
            fh >> token;   
            data_read.push_back( token);
        }

        // the last element is read twice
        data_read.pop_back();

        // check if reading result is consistent
        if( ! fh.eof()) 
        {
            if (verbose)
                std::cerr << "WARNING : readData() : reading file might have failed." 
                << std::endl;
        }

        fh.close();

        // check if the given handle is already initialized
        if( NULL != *data) 
        {
            if( *len != data_read.size()) 
            {
                if (verbose)
                    std::cerr << "cutReadFile() : Initialized memory given but size "
                    << " mismatch with signal read." << std::endl;

                return CUTFalse;
            }
        }
        else 
        {
            // allocate storage for the data read
			*data = (T*) malloc( sizeof(T) * data_read.size());
            // store signal size
            *len = static_cast<unsigned int>( data_read.size());
        }

        // copy data
        memcpy( *data, &data_read.front(), sizeof(T) * data_read.size());

        return CUTTrue;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename 
    //! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
    //! @param filename name of the source file
    //! @param data  data to write
    //! @param len  number of data elements in data, -1 on error
    //! @param epsilon  epsilon for comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    cutWriteFile( const char* filename, const T* data, unsigned int len,
                  const T epsilon, bool verbose) 
    {
        CUT_CONDITION( NULL != filename);
        CUT_CONDITION( NULL != data);

        // open file for writing
        std::fstream fh( filename, std::fstream::out);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Opening file failed." << std::endl;
            return CUTFalse;
        }

        // first write epsilon
        fh << "# " << epsilon << "\n";

        // write data
        for( unsigned int i = 0; (i < len) && (fh.good()); ++i) 
        {
            fh << data[i] << ' ';
        }

        // Check if writing succeeded
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Writing file failed." << std::endl;
            return CUTFalse;
        }

        // file ends with nl
        fh << std::endl;

        return CUTTrue;
    }

} // end, namespace unnamed
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//! Deallocate memory allocated within Cutil
//! @param  pointer to memory 
//////////////////////////////////////////////////////////////////////////////
void CUTIL_API
cutFree( void* ptr) {
  if( NULL != ptr) free( ptr);
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are either in data/ or in ../../../projects/<executable_name>/data/.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char* CUTIL_API
cutFindFilePath(const char* filename, const char* executable_path) 
{
    // search in data/
    if (filename == 0)
        return 0;
    size_t filename_len = strlen(filename);
    const char data_folder[] = "data/";
    size_t data_folder_len = strlen(data_folder);
    char* file_path = 
      (char*) malloc( sizeof(char) * (data_folder_len + filename_len + 1));
    strcpy(file_path, data_folder);
    strcat(file_path, filename);
    std::fstream fh0(file_path, std::fstream::in);
    if (fh0.good())
        return file_path;
    free( file_path);

    // search in ../../../projects/<executable_name>/data/
    if (executable_path == 0)
        return 0;
    size_t executable_path_len = strlen(executable_path);
    const char* exe;
    for (exe = executable_path + executable_path_len - 1; 
         exe >= executable_path; --exe)
        if (*exe == '/' || *exe == '\\')
            break;
    if (exe < executable_path)
        exe = executable_path;
    else
        ++exe;
    size_t executable_len = strlen(exe);
    size_t executable_dir_len = executable_path_len - executable_len;
    const char projects_relative_path[] = "../../../projects/";
    size_t projects_relative_path_len = strlen(projects_relative_path);
    file_path = 
      (char*) malloc( sizeof(char) * (executable_path_len +
         projects_relative_path_len + 1 + data_folder_len + filename_len + 1));
    strncpy(file_path, executable_path, executable_dir_len);
    file_path[executable_dir_len] = '\0';
    strcat(file_path, projects_relative_path);
    strcat(file_path, exe);
    size_t file_path_len = strlen(file_path);
    if (*(file_path + file_path_len - 1) == 'e' &&
        *(file_path + file_path_len - 2) == 'x' &&
        *(file_path + file_path_len - 3) == 'e' &&
        *(file_path + file_path_len - 4) == '.') {
        *(file_path + file_path_len - 4) = '/';
        *(file_path + file_path_len - 3) = '\0';
    }
    else {
        *(file_path + file_path_len - 0) = '/';
        *(file_path + file_path_len + 1) = '\0';
    }
    strcat(file_path, data_folder);
    strcat(file_path, filename);
    std::fstream fh1(file_path, std::fstream::in);
    if (fh1.good())
        return file_path;
    free( file_path);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg single precision floating point data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFilef( const char* filename, float** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg double precision floating point data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFiled( const char* filename, double** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFilei( const char* filename, int** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFileui( const char* filename, unsigned int** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}


////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg char / byte data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFileb( const char* filename, char** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned char / byte data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutReadFileub( const char* filename, unsigned char** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for single precision floating point data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFilef( const char* filename, const float* data, unsigned int len,
               const float epsilon, bool verbose) 
{
    return cutWriteFile( filename, data, len, epsilon, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for double precision floating point data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFiled( const char* filename, const double* data, unsigned int len,
               const double epsilon, bool verbose) 
{
    return cutWriteFile( filename, data, len, epsilon, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for integer data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFilei( const char* filename, const int* data, unsigned int len, bool verbose) 
{
    return cutWriteFile( filename, data, len, 0, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned integer data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFileui( const char* filename,const unsigned int* data,unsigned int len, bool verbose)
{
    return cutWriteFile( filename, data, len, static_cast<unsigned int>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFileb( const char* filename, const char* data, unsigned int len, bool verbose) 
{  
    return cutWriteFile( filename, data, len, static_cast<char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFileub( const char* filename, const unsigned char* data, 
                unsigned int len, bool verbose) 
{  
    return cutWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutWriteFileb( const char* filename,const unsigned char* data,unsigned int len, bool verbose)
{
    return cutWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if command line argument \a flag-name is given
//! @return 1 if command line argument \a flag_name has been given, otherwise 0
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param flag_name  name of command line flag
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCheckCmdLineFlag( const int argc, const char** argv, const char* flag_name) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initalize 
        CmdArgReader::init( argc, argv);

        // check if the command line argument exists
        if( CmdArgReader::existArg( flag_name)) 
        {
            ret_val = CUTTrue;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type int
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutGetCmdLineArgumenti( const int argc, const char** argv, 
                        const char* arg_name, int* val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const int* v = CmdArgReader::getArg<int>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = CUTTrue;
        }		
		else {
			// fail safe
			val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type float
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutGetCmdLineArgumentf( const int argc, const char** argv, 
                       const char* arg_name, float* val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const float* v = CmdArgReader::getArg<float>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = CUTTrue;
        }
		else {
			// fail safe
			val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type string
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutGetCmdLineArgumentstr( const int argc, const char** argv, 
                         const char* arg_name, char** val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const std::string* v = CmdArgReader::getArg<std::string>( arg_name);
        if( NULL != v) 
        {

            // allocate memory for the string
            *val = (char*) malloc( sizeof(char) * (v->length() + 1));
            // copy from string to c_str
            strcpy( *val, v->c_str());
            ret_val = CUTTrue;
        }		
		else {
			// fail safe
			*val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string."<< 
        std::endl;
    } 

    return ret_val;

}

////////////////////////////////////////////////////////////////////////////////
//! Extended assert
//! @return CUTTrue if the condition \a val holds, otherwise CUTFalse
//! @param val  condition to test
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCheckCondition( int val, const char* file, const int line) 
{
    CUTBoolean ret_val = CUTTrue;

    try 
    {
        // check for error
        ErrorChecker::condition( (0 == val) ? false : true, file, line);
    }
    catch( const std::exception& ex) 
    {
        // print where the exception occured
        std::cerr << ex.what() << std::endl;
        ret_val = CUTFalse;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutComparef( const float* reference, const float* data,
            const unsigned int len ) 
{
    const float epsilon = 0.0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutComparei( const int* reference, const int* data,
            const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCompareub( const unsigned char* reference, const unsigned char* data,
             const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutComparefe( const float* reference, const float* data,
             const unsigned int len, const float epsilon ) 
{
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for equality
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    CUT_CONDITION( epsilon >= 0);

    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return CUTFalse;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG
    if( ! result) 
    {
        std::cerr << "ERROR, l2-norm error " 
            << error << " is greater than epsilon " << epsilon << "\n";
    }
#endif

    return result ? CUTTrue : CUTFalse;
}

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return CUTTrue if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutCreateTimer( unsigned int* name) 
{
    *name = StopWatch::create();

    return (0 != name) ? CUTTrue : CUTFalse;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return CUTTrue if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutDeleteTimer( unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

    try 
    {
        StopWatch::destroy( name);
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutStartTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).start();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).start();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutStopTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).stop();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).stop();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API
cutResetTimer( const unsigned int name)
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).reset();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).reset();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer 
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
float CUTIL_API
cutGetAverageTimerValue( const unsigned int name)
{
    float time = 0.0;

#ifdef _DEBUG
    try 
    {
        time = StopWatch::get( name).getAverageTime();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
    }
#else
    time = StopWatch::get( name).getAverageTime();
#endif

    return time;
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
float CUTIL_API
cutGetTimerValue( const unsigned int name) 
{  
    float time = 0.0;

#ifdef _DEBUG
    try 
    {
        time = StopWatch::get( name).getTime();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
    }
#else
    time = StopWatch::get( name).getTime();
#endif

    return time;
}



////////////////////////////////////////////////////////////////////////////////
//! TBD
////////////////////////////////////////////////////////////////////////////////
void CUTIL_API
cutCheckBankAccess( unsigned int tidx, unsigned int tidy, unsigned int tidz,
                   unsigned int bdimx, unsigned int bdimy, unsigned int bdimz,
                   const char* file, const int line, const char* aname,
                   const int index) 
{
    BankChecker::getHandle()->access( tidx, tidy, tidz, 
                                      bdimx, bdimy, bdimz,
                                      file, line, aname, index );
}

