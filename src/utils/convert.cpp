/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/util.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <boost/algorithm/string.hpp>

MTS_NAMESPACE_BEGIN

class Convert : public Utility {
public:
	void convert(const std::string &s1, const std::string &s2) {
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();

		fs::path iFile = fileResolver->resolve(s1);
		ref<FileStream> is = new FileStream(s1, FileStream::EReadOnly);

		fs::path oFile = fileResolver->resolve(s2);
		ref<Bitmap> iBitmap = new Bitmap(Bitmap::EAuto, is);
		iBitmap->write(oFile);
	}

	int run(int argc, char **argv) {
		if (argc < 3) {
			cout << "Convert image format" << endl;
			cout << "convert <input_file> <output_file>" << endl;
		} else {
			convert(argv[1], argv[2]);
		}
		return 0;
	}

	MTS_DECLARE_UTILITY()
};

MTS_EXPORT_UTILITY(Convert, "Convert image format");
MTS_NAMESPACE_END
