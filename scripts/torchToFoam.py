"""
Script for writing predicted b directly into a foam file.
Functionality currently lies in writeToFoam notebook and will be ported here when done.
"""

import os, sys
import torch as th

# own scrips
sys.path.insert(1, '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/')
import preProcess as pre


# dict for boundary types
b_type = {'empty': "        {0:<16}{1}\n".format("type", "empty;"),
          'fixedValue_uniform': "        {0:<16}{1}\n".format("type", "fixedValue;")+
                                "        {0:<16}{1}\n".format("value", "uniform (0 0 0 0 0 0);"),
          'zeroGradient': "        {0:<16}{1}\n".format("type", "zeroGradient;"),
          'symmetryPlane':"        {0:<16}{1}\n".format("type", "symmetryPlane;"),
          'cyclic': "        {0:<16}{1}\n".format("type", "cyclic;"),
          'fixedValue_nonuniform': "        {0:<16}{1}\n".format("type", "fixedValue;")+
                                   "        {0:<16}{1}\n".format("value", "nonuniform List<symmTensor>")}


def writesymmtensor(tensor,
                    filename,
                    boundaries):

    n = tensor.shape[0]

    # read in header
    hf = open('/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/foam_header', 'r')
    if hf.mode == 'r':
        header = hf.read()
    hf.close()

    # open file to write
    of = open(filename, 'w')

    # check if mode is write
    if of.mode == 'w':

        # write header
        of.write(header)

        # write number of internal points
        of.write("{}\n(\n".format(tensor.shape[0]))

        # write internal points
        for point in tensor:
            of.write("({:9f} {:9f} {:9f} {:9f} {:9f} {:9f})\n".format(point[0, 0], point[0, 1],
                                                                      point[0, 2], point[1, 1],
                                                                      point[1, 2], point[2, 2]))

        # write boundary patches
        of.write(")\n;\n\nboundaryField\n{\n")

        # loop over boundaries list
        for boundary in boundaries:
            of.write("    {}\n".format(boundary[0]))
            of.write("    {\n")
            of.write(b_type[boundary[1]])

            # write boundary filed if specified
            if boundary[1] == 'fixedValue_nonuniform':
                of.write("{}\n(\n".format(boundary[2].shape[0]))
                for b_point in boundary[2]:
                    of.write("({:9f} {:9f} {:9f} {:9f} {:9f} {:9f})\n".format(b_point[0, 0], b_point[0, 1],
                                                                              b_point[0, 2], b_point[1, 1],
                                                                              b_point[1, 2], b_point[2, 2]))
                of.write(")\n;\n")
            # close boundary bracket
            of.write("    }\n")

        of.write("}")

        # write closing lines of foam
        of.write("\n\n\n// ************************************************************************* //")

    # close file
    of.close()


if __name__=='__main__':

    # set b
    tmp = th.rand(10, 3, 3)
    b0 = 0.5 * (tmp + tmp.transpose(1, 2))

    # set boundary b
    tmp = th.rand(5, 3, 3)
    boundary_b = 0.5 * (tmp + tmp.transpose(1, 2))

    # list for boundary names and corresponding type
    b_list = [('inlet', 'fixedValue_nonuniform', boundary_b),
              ('outlet', 'zeroGradient'),
              ('upperWall', 'fixedValue_uniform'),
              ('lowerWall', 'fixedValue_uniform'),
              ('frontAndBack', 'empty')]

    # write tensor to file
    writesymmtensor(b0, '/home/leonriccius/Desktop/b_dd', b_list)
