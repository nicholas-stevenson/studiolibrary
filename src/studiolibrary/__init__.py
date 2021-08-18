# Copyright 2020 by Kurt Rathjen. All Rights Reserved.
#
# This library is free software: you can redistribute it and/or modify it 
# under the terms of the GNU Lesser General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. This library is distributed in the 
# hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

__version__ = "2.9.6.b3"
__pose_version__ = "1.1.0"


def version():
    """
    Return the current version of the Studio Library

    :rtype: str
    """
    return __version__


def poseVersion():
    """
    Returns the most recent Pose Version

    :rtype: str

    """
    return __pose_version__


def poseVersionSummary():
    msg = list()
    msg.append("Latest Pose Version: 1.1.0")

    msg.append("\n")

    msg.append("1.1.0")
    msg.append("These contain additional Matrix data and are compatible")
    msg.append("with Animation Layers and Relative To posing.")

    msg.append("\n")

    msg.append("1.0.0")
    msg.append("This is the base Studio Library pose version.")

    return '\n'.join(msg)


from studiolibrary import config
from studiolibrary import resource
from studiolibrary.utils import *
from studiolibrary.library import Library
from studiolibrary.libraryitem import LibraryItem
from studiolibrary.main import main
