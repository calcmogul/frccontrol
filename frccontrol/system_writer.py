"""A class that handles writing out matrices of a system to a C++ or Java file.
"""

import numpy as np
import os


class SystemWriter:
    def __init__(
        self,
        system,
        class_name,
        header_path_prefix,
        header_extension,
        period_variant=False,
    ):
        """Exports matrices to pair of C++ source files.

        Keyword arguments:
        system -- System object
        class_name -- subsystem class name in camel case
        header_path_prefix -- path prefix in which header exists
        header_extension -- file extension of header file
        period_variant -- True to use PeriodVariantLoop, False to use
                          StateSpaceLoop
        """
        self.system = system
        self.class_name = class_name
        self.header_path_prefix = header_path_prefix
        self.header_extension = header_extension
        template_cpp = (
            "<"
            + str(system.sysd.A.shape[0])
            + ", "
            + str(system.sysd.B.shape[1])
            + ", "
            + str(system.sysd.C.shape[0])
            + ">"
        )

        template_java = (
            "<N"
            + str(system.sysd.A.shape[0])
            + ", N"
            + str(system.sysd.B.shape[1])
            + ", N"
            + str(system.sysd.C.shape[0])
            + ">"
        )

        self.period_variant = period_variant
        if period_variant:
            self.class_type = "PeriodVariant"
            self.plant_coeffs_header = "PeriodVariantPlantCoeffs"
            self.obsv_coeffs_header = "PeriodVariantObserverCoeffs"
            self.loop_header = "PeriodVariantLoop"
        else:
            self.class_type = "StateSpace"
            self.plant_coeffs_header = "StateSpacePlantCoeffs"
            self.obsv_coeffs_header = "StateSpaceObserverCoeffs"
            self.loop_header = "StateSpaceLoop"

        self.ctrl_coeffs_header = "StateSpaceControllerCoeffs"
        self.ctrl_coeffs_type_cpp = "frc::" + self.ctrl_coeffs_header + template_cpp
        self.plant_coeffs_type_cpp = "frc::" + self.plant_coeffs_header + template_cpp
        self.obsv_coeffs_type_cpp = "frc::" + self.obsv_coeffs_header + template_cpp
        self.loop_type_cpp = "frc::" + self.loop_header + template_cpp

        self.ctrl_coeffs_type_java = self.ctrl_coeffs_header + template_java
        self.plant_coeffs_type_java = self.plant_coeffs_header + template_java
        self.obsv_coeffs_type_java = self.obsv_coeffs_header + template_java
        self.loop_type_java = self.loop_header + template_java

    def write_cpp_header(self):
        """Writes C++ header file."""
        prefix = "#include <frc/controller/"
        headers = []
        headers.append(prefix + self.plant_coeffs_header + ".h>")
        headers.append(prefix + self.ctrl_coeffs_header + ".h>")
        headers.append(prefix + self.obsv_coeffs_header + ".h>")
        headers.append(prefix + self.loop_header + ".h>")

        with open(
            self.class_name + "Coeffs." + self.header_extension, "w"
        ) as header_file:
            print("#pragma once" + os.linesep, file=header_file)
            for header in sorted(headers):
                print(header, file=header_file)
            header_file.write(os.linesep)
            self.__write_cpp_func_name(
                header_file, self.plant_coeffs_type_cpp, "PlantCoeffs", in_header=True
            )
            self.__write_cpp_func_name(
                header_file,
                self.ctrl_coeffs_type_cpp,
                "ControllerCoeffs",
                in_header=True,
            )
            self.__write_cpp_func_name(
                header_file, self.obsv_coeffs_type_cpp, "ObserverCoeffs", in_header=True
            )
            self.__write_cpp_func_name(
                header_file, self.loop_type_cpp, "Loop", in_header=True
            )

    def write_cpp_source(self):
        """Writes C++ source file."""
        if len(self.header_path_prefix) > 0 and self.header_path_prefix[-1] != os.sep:
            self.header_path_prefix += os.sep

        with open(self.class_name + "Coeffs.cpp", "w") as source_file:
            print(
                '#include "'
                + self.header_path_prefix
                + self.class_name
                + "Coeffs."
                + self.header_extension
                + '"'
                + os.linesep,
                file=source_file,
            )
            print("#include <Eigen/Core>" + os.linesep, file=source_file)

            # Write MakePlantCoeffs()
            self.__write_cpp_func_name(
                source_file, self.plant_coeffs_type_cpp, "PlantCoeffs", in_header=False
            )
            if self.period_variant:
                self.__write_cpp_matrix(source_file, self.system.sysc.A, "Acontinuous")
                self.__write_cpp_matrix(source_file, self.system.sysc.B, "Bcontinuous")
                self.__write_cpp_matrix(source_file, self.system.sysd.C, "C")
                self.__write_cpp_matrix(source_file, self.system.sysd.D, "D")
                print(
                    "  return "
                    + self.plant_coeffs_type_cpp
                    + "(Acontinuous, Bcontinuous, C, D);",
                    file=source_file,
                )
            else:
                self.__write_cpp_matrix(source_file, self.system.sysd.A, "A")
                self.__write_cpp_matrix(source_file, self.system.sysd.B, "B")
                self.__write_cpp_matrix(source_file, self.system.sysd.C, "C")
                self.__write_cpp_matrix(source_file, self.system.sysd.D, "D")
                print(
                    "  return " + self.plant_coeffs_type_cpp + "(A, B, C, D);",
                    file=source_file,
                )
            print("}" + os.linesep, file=source_file)

            # Write MakeControllerCoeffs()
            self.__write_cpp_func_name(
                source_file,
                self.ctrl_coeffs_type_cpp,
                "ControllerCoeffs",
                in_header=False,
            )
            self.__write_cpp_matrix(source_file, self.system.K, "K")
            self.__write_cpp_matrix(source_file, self.system.Kff, "Kff")
            self.__write_cpp_matrix(source_file, self.system.u_min, "Umin")
            self.__write_cpp_matrix(source_file, self.system.u_max, "Umax")
            print(
                "  return " + self.ctrl_coeffs_type_cpp + "(K, Kff, Umin, Umax);",
                file=source_file,
            )
            print("}" + os.linesep, file=source_file)

            # Write MakeObserverCoeffs()
            self.__write_cpp_func_name(
                source_file,
                self.obsv_coeffs_type_cpp,
                "ObserverCoeffs",
                in_header=False,
            )
            if self.period_variant:
                self.__write_cpp_matrix(source_file, self.system.Q, "Qcontinuous")
                self.__write_cpp_matrix(source_file, self.system.R, "Rcontinuous")
                self.__write_cpp_matrix(
                    source_file, self.system.P_steady, "PsteadyState"
                )

                first_line_prefix = "  return " + self.obsv_coeffs_type_cpp + "("
                space_prefix = " " * len(first_line_prefix)
                print(first_line_prefix + "Qcontinuous, Rcontinuous,", file=source_file)
                print(space_prefix + "PsteadyState);", file=source_file)
            else:
                self.__write_cpp_matrix(source_file, self.system.kalman_gain, "K")
                print(
                    "  return " + self.obsv_coeffs_type_cpp + "(K);", file=source_file
                )
            print("}" + os.linesep, file=source_file)

            # Write MakeLoop()
            self.__write_cpp_func_name(
                source_file, self.loop_type_cpp, "Loop", in_header=False
            )
            first_line_prefix = "  return " + self.loop_type_cpp + "("
            space_prefix = " " * len(first_line_prefix)
            print(
                first_line_prefix + "Make" + self.class_name + "PlantCoeffs(),",
                file=source_file,
            )
            print(
                space_prefix + "Make" + self.class_name + "ControllerCoeffs(),",
                file=source_file,
            )
            print(
                space_prefix + "Make" + self.class_name + "ObserverCoeffs());",
                file=source_file,
            )
            print("}", file=source_file)

    def write_java_source(self):
        wpilibj_prefix = "import edu.wpi.first.wpilibj."
        wpiutil_prefix = "import edu.wpi.first.wpiutil."
        imports = []

        # State space types
        imports.append(wpilibj_prefix + "controller." + self.plant_coeffs_header + ";")
        imports.append(wpilibj_prefix + "controller." + self.ctrl_coeffs_header + ";")
        imports.append(wpilibj_prefix + "controller." + self.obsv_coeffs_header + ";")
        imports.append(wpilibj_prefix + "controller." + self.loop_header + ";")

        # Number types and matrices
        imports.append(wpiutil_prefix + "math.numbers.*;")
        imports.append(wpiutil_prefix + "math.*;")

        with open(self.class_name + "Coeffs.java", "w") as source_file:
            for imp in sorted(imports):
                print(imp, file=source_file)

            source_file.write(os.linesep)

            print("public class " + self.class_name + "Coeffs {", file=source_file)

            states = "Nat.N" + str(self.system.sysd.A.shape[0]) + "()"
            inputs = "Nat.N" + str(self.system.sysd.B.shape[1]) + "()"
            outputs = "Nat.N" + str(self.system.sysd.C.shape[0]) + "()"
            # MakePlantCoeffs()
            self.__write_java_func_name(
                source_file, self.plant_coeffs_type_java, "PlantCoeffs"
            )
            if self.period_variant:
                self.__write_java_matrix(source_file, self.system.sysc.A, "Acontinuous")
                self.__write_java_matrix(source_file, self.system.sysc.B, "Bcontinuous")
                self.__write_java_matrix(source_file, self.system.sysd.C, "C")
                self.__write_java_matrix(source_file, self.system.sysd.D, "D")

                print(
                    "    return "
                    + self.plant_coeffs_type_java
                    + "("
                    + states
                    + ", "
                    + inputs
                    + ", "
                    + outputs
                    + ", Acontinuous, Bcontinuous, C, D);",
                    file=source_file,
                )
            else:
                self.__write_java_matrix(source_file, self.system.sysd.A, "A")
                self.__write_java_matrix(source_file, self.system.sysd.B, "B")
                self.__write_java_matrix(source_file, self.system.sysd.C, "C")
                self.__write_java_matrix(source_file, self.system.sysd.D, "D")

                print(
                    "    return new "
                    + self.plant_coeffs_type_java
                    + "("
                    + states
                    + ", "
                    + inputs
                    + ", "
                    + outputs
                    + ", A, B, C, D);",
                    file=source_file,
                )
            print("  }" + os.linesep, file=source_file)

            # MakeControllerCoeffs()
            self.__write_java_func_name(
                source_file, self.ctrl_coeffs_type_java, "ControllerCoeffs"
            )
            self.__write_java_matrix(source_file, self.system.K, "K")
            self.__write_java_matrix(source_file, self.system.Kff, "Kff")
            self.__write_java_matrix(source_file, self.system.u_min, "Umin")
            self.__write_java_matrix(source_file, self.system.u_max, "Umax")
            print(
                "    return new "
                + self.ctrl_coeffs_type_java
                + "(K, Kff, Umin, Umax);",
                file=source_file,
            )
            print("  }" + os.linesep, file=source_file)

            # MakeObserverCoeffs()
            self.__write_java_func_name(
                source_file, self.obsv_coeffs_type_java, "ObserverCoeffs"
            )
            if self.period_variant:
                self.__write_java_matrix(source_file, self.system.Q, "Qcontinuous")
                self.__write_java_matrix(source_file, self.system.R, "Rcontinuous")
                self.__write_java_matrix(
                    source_file, self.system.P_steady, "PsteadyState"
                )

                first_line_prefix = "    return new " + self.obsv_coeffs_type_java + "("
                space_prefix = " " * len(first_line_prefix)
                print(first_line_prefix + "Qcontinuous, Rcontinuous,", file=source_file)
                print(space_prefix + "PsteadyState);", file=source_file)
            else:
                self.__write_java_matrix(source_file, self.system.kalman_gain, "K")
                print(
                    "    return new " + self.obsv_coeffs_type_java + "(K);",
                    file=source_file,
                )
            print("  }" + os.linesep, file=source_file)

            # Write MakeLoop()
            self.__write_java_func_name(source_file, self.loop_type_java, "Loop")
            first_line_prefix = "    return new " + self.loop_type_java + "("
            space_prefix = " " * len(first_line_prefix)
            print(
                first_line_prefix + "make" + self.class_name + "PlantCoeffs(),",
                file=source_file,
            )
            print(
                space_prefix + "make" + self.class_name + "ControllerCoeffs(),",
                file=source_file,
            )
            print(
                space_prefix + "make" + self.class_name + "ObserverCoeffs());",
                file=source_file,
            )

            print("  }", file=source_file)
            print("}", file=source_file)

    def __write_cpp_func_name(self, cpp_file, return_type, object_suffix, in_header):
        """Writes either declaration or definition of C++ factory function.

        Keyword arguments:
        cpp_file -- file object to which to write name
        return_type -- string containing the return type
        object_suffix -- asdf
        in_header -- if True, print prototype instead of declaration
        """
        if in_header:
            func_suffix = ";"
        else:
            func_suffix = " {"
        func_name = "Make" + self.class_name + object_suffix + "()" + func_suffix
        if len(return_type + " " + func_name) > 80:
            print(return_type, file=cpp_file)
            print(func_name, file=cpp_file)
        else:
            print(return_type + " " + func_name, file=cpp_file)

    def __write_cpp_matrix(self, cpp_file, matrix, matrix_name):
        print(
            "  Eigen::Matrix<double, "
            + str(matrix.shape[0])
            + ", "
            + str(matrix.shape[1])
            + "> "
            + matrix_name
            + ";",
            file=cpp_file,
        )
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                print(
                    "  "
                    + matrix_name
                    + "("
                    + str(row)
                    + ", "
                    + str(col)
                    + ") = "
                    + str(matrix[row, col])
                    + ";",
                    file=cpp_file,
                )

    def __write_java_func_name(self, java_file, return_type, object_suffix):
        func_name = "make" + self.class_name + object_suffix + "() {"
        if len("  public static " + return_type + " " + func_name) > 80:
            print("  public static " + return_type, file=java_file)
            print("    " + func_name, file=java_file)
        else:
            print("  public static " + return_type + " " + func_name, file=java_file)

    def __write_java_matrix(self, java_file, matrix, matrix_name):
        row_major_data = np.ravel(matrix)
        # Matrix<R, C> <name> = MatrixUtils.mat(R, C).fill(
        decl = (
            "    Matrix<N"
            + str(matrix.shape[0])
            + ", N"
            + str(matrix.shape[1])
            + "> "
            + matrix_name
            + " = MatrixUtils.mat(Nat.N"
            + str(matrix.shape[0])
            + "(), Nat.N"
            + str(matrix.shape[1])
            + "()).fill("
        )

        for (i, value) in enumerate(row_major_data):
            decl += str(value)
            if i != matrix.shape[0] * matrix.shape[1] - 1:
                decl += ", "

        decl += ");"

        print(decl, file=java_file)
