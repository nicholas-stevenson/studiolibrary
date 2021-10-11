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
"""
#
# pose.py
import mutils

# Example 1:
# Save and load a pose from the selected objects
objects = maya.cmds.ls(selection=True)
mutils.savePose("/tmp/pose.json", objects)

mutils.loadPose("/tmp/pose.json")

# Example 2:
# Create a pose object from a list of object names
pose = mutils.Pose.fromObjects(objects)

# Example 3:
# Create a pose object from the selected objects
objects = maya.cmds.ls(selection=True)
pose = mutils.Pose.fromObjects(objects)

# Example 4:
# Save the pose object to disc
path = "/tmp/pose.json"
pose.save(path)

# Example 5:
# Create a pose object from disc
path = "/tmp/pose.json"
pose = mutils.Pose.fromPath(path)

# Load the pose on to the objects from file
pose.load()

# Load the pose to the selected objects
objects = maya.cmds.ls(selection=True)
pose.load(objects=objects)

# Load the pose to the specified namespaces
pose.load(namespaces=["character1", "character2"])

# Load the pose to the specified objects
pose.load(objects=["Character1:Hand_L", "Character1:Finger_L"])

"""
import logging

import msn.maya.rig.query
import msn.maya.rig.types.character
import msn.maya.rig.components.ik
import mutils
import shared.maya.api.matrix
import shared.maya.api.object
import shared.maya.decorators
import shared.maya.namespace
import shared.python.math
import shared.maya.hierarchy
import shared.maya.animation.cache
import shared.maya.attribute
import copy

try:
    import maya.cmds
    import maya.api.OpenMaya as om2
except ImportError:
    import traceback

    traceback.print_exc()

__all__ = ["Pose", "savePose", "loadPose"]

logger = logging.getLogger(__name__)

_pose_ = None


def savePose(path, objects, metadata=None):
    """
    Convenience function for saving a pose to disc for the given objects.

    Example:
        path = "C:/example.pose"
        pose = savePose(path, metadata={'description': 'Example pose'})
        print(pose.metadata())
        # {
        'user': 'Hovel', 
        'mayaVersion': '2016', 
        'description': 'Example pose'
        }

    :type path: str
    :type objects: list[str]
    :type metadata: dict or None
    :rtype: Pose
    """
    pose = mutils.Pose.fromObjects(objects)

    if metadata:
        pose.updateMetadata(metadata)

    pose.save(path)

    return pose


def loadPose(path, *args, **kwargs):
    """
    Convenience function for loading the given pose path.
    
    :type path: str
    :type args: list
    :type kwargs: dict 
    :rtype: Pose 
    """
    global _pose_

    clearCache = kwargs.get("clearCache")

    if not _pose_ or _pose_.path() != path or clearCache:
        _pose_ = Pose.fromPath(path)

    _pose_.load(*args, **kwargs)

    return _pose_


class Pose(mutils.TransferObject):

    def __init__(self):
        mutils.TransferObject.__init__(self)

        self._cache = None
        self._mtime = None
        self._cacheKey = None
        self._isLoading = False
        self._selection = None
        self._mirrorTable = None
        self._autoKeyFrame = None

        # Pubg internal variables
        self._rig_list = list()
        self._ik_controllers = dict()

        # Temporary variables used when applying a pose relative to another another node
        self._relative_to_snapshot = dict()
        self._gun_relative_to_snapshot = dict()

    def createObjectData(self, name):
        """
        Create the object data for the given object name.
        
        :type name: str
        :rtype: dict
        """
        attrs = maya.cmds.listAttr(name, unlocked=True, keyable=True) or []

        if self.hasTransforms(attrs):
            attrs.extend(["matrix", "worldMatrix"])

        attrs = list(set(attrs))
        attrs = [mutils.Attribute(name, attr) for attr in attrs]

        data = {"attrs": self.attrs(name)}

        for attr in attrs:
            if attr.isValid():
                if attr.value() is None:
                    msg = "Cannot save the attribute %s with value None."
                    logger.warning(msg, attr.fullname())
                else:
                    value = attr.value()
                    data["attrs"][attr.attr()] = {
                        "type":  attr.type(),
                        "value": value
                    }

        return data

    def select(self, objects=None, namespaces=None, **kwargs):
        """
        Select the objects contained in the pose file.
        
        :type objects: list[str] or None
        :type namespaces: list[str] or None
        :rtype: None
        """
        selectionSet = mutils.SelectionSet.fromPath(self.path())
        selectionSet.load(objects=objects, namespaces=namespaces, **kwargs)

    def cache(self):
        """
        Return the current cached attributes for the pose.
        
        :rtype: list[(Attribute, Attribute)]
        """
        return self._cache

    def attrs(self, name):
        """
        Return the attribute for the given name.
        
        :type name: str
        :rtype: dict
        """
        return self.object(name).get("attrs", {})

    def attr(self, name, attr):
        """
        Return the attribute data for the given name and attribute.

        :type name: str
        :type attr: str
        :rtype: dict
        """
        return self.attrs(name).get(attr, {})

    def attrType(self, name, attr):
        """
        Return the attribute type for the given name and attribute.
        
        :type name: str
        :type attr: str
        :rtype: str
        """
        return self.attr(name, attr).get("type", None)

    def attrValue(self, name, attr):
        """
        Return the attribute value for the given name and attribute.
        
        :type name: str
        :type attr: str
        :rtype: str | int | float
        """
        return self.attr(name, attr).get("value", None)

    def setMirrorAxis(self, name, mirrorAxis):
        """
        Set the mirror axis for the given name.
        
        :type name: str
        :type mirrorAxis: list[int]
        """
        if name in self.objects():
            self.object(name).setdefault("mirrorAxis", mirrorAxis)
        else:
            msg = "Object does not exist in pose. " \
                  "Cannot set mirror axis for %s"

            logger.debug(msg, name)

    def mirrorAxis(self, name):
        """
        Return the mirror axis for the given name.
        
        :rtype: list[int] | None
        """
        result = None
        if name in self.objects():
            result = self.object(name).get("mirrorAxis", None)

        if result is None:
            logger.debug("Cannot find mirror axis in pose for %s", name)

        return result

    def updateMirrorAxis(self, name, mirrorAxis):
        """
        Update the mirror axis for the given object name.
        
        :type name: str
        :type mirrorAxis: list[int]
        """
        self.setMirrorAxis(name, mirrorAxis)

    def mirrorTable(self):
        """
        Return the Mirror Table for the pose.
        
        :rtype: mutils.MirrorTable
        """
        return self._mirrorTable

    def setMirrorTable(self, mirrorTable):
        """
        Set the Mirror Table for the pose.
        
        :type mirrorTable: mutils.MirrorTable
        """
        objects = self.objects().keys()
        self._mirrorTable = mirrorTable

        for srcName, dstName, mirrorAxis in mirrorTable.matchObjects(objects):
            self.updateMirrorAxis(dstName, mirrorAxis)

    def mirrorValue(self, name, attr, mirrorAxis):
        """
        Return the mirror value for the given name, attribute and mirror axis.
        
        :type name: str
        :type attr: str
        :type mirrorAxis: list[]
        :rtype: None | int | float
        """
        value = None

        if self.mirrorTable() and name:

            value = self.attrValue(name, attr)

            if value is not None:
                value = self.mirrorTable().formatValue(attr, value, mirrorAxis)
            else:
                logger.debug("Cannot find mirror value for %s.%s", name, attr)

        return value

    def updateRigList(self):
        """
        Based on the selected namespaces, generate a list of native rig objects.
        This internal list is destroyed after the pose has finished applying.
        """
        if not self._isLoading:
            for namespace in self.namespaces():
                rig = msn.maya.rig.query.get_rig(namespace)
                if rig:
                    self._rig_list.append(rig)

    def resetRigList(self):
        self._rig_list = list()
        self._ik_controllers = dict()

    def captureRigStates(self):
        for rig in self._rig_list:
            rig.snapshot_rig_states()

    def restoreRigStates(self):
        for rig in self._rig_list:
            rig.restore_rig_states()

    def setRigsToPosing(self, keyframe=False):
        for rig in self._rig_list:
            for ik_system in rig.ik_systems:
                if not ik_system.ik_state or not self.isIkSystemPosing(ik_system):
                    continue

                if keyframe:
                    ik_system.key()

                ik_system.ik_state = False

    def appendFkSystems(self, objects):
        """
        Studio Library does not pose IK controllers and instead poses the FK bones.  But if an animator selects only an IK controller and
        attempts to pose an arm or leg, Studio Library will silently do nothing as it will ignore the posing of this selected IK controller.

        So in the event that we have ik systems selected, ensure that the object list also includes any FK bones that are part of that same system.
        """
        for rig in self._rig_list:
            for ik_system in rig.ik_systems:
                if ik_system.ik_state and any([ik_system.control in objects, ik_system.pole_vector_control in objects]):
                    for bone in ik_system.bone_list:
                        if bone not in objects:
                            objects.append(bone)

    def loadIkTwistValues(self, keyframe):
        """
        ik twist and corrective offsets are a temporary fix, exposed on the rig's limb ik controllers,
        which allow animators to manually inject offset values into both the twist bones,
        as well as the "twist" motion of the upper arms and forearms.  This is a legacy issue from an animation rig
        generated in Motion Builder, where instead of treating limbs like a single plane lever, actually twisted the bones
        on themselves.

        Also for the twist bones, this was because the twists were broken in many scenes of the motion builder rig,
        so some animations have twist values, others don't, and others have weird values baked in.

        So while we want to actually pose IK controllers and instead want to leave that to the animation rig to switch
        between IK and FK, we need these stored values from the pose.
        """

        arm_twist_attributes = ["upperarm_twist",
                                "upperarm_corrective_offset",
                                "forearm_twist",
                                "wrist_twist_offset"]

        leg_twist_attributes = ["hip_twist",
                                "hip_corrective_offset",
                                "calf_twist",
                                "calf_twist_offset"]

        for rig in self._rig_list:
            for ik_system in rig.ik_systems:
                if ik_system.ik_state and ik_system.control in self._ik_controllers.keys():

                    if isinstance(ik_system, msn.maya.rig.components.ik.ArmIKSystem):
                        ik_twist_attributes = arm_twist_attributes
                    elif isinstance(ik_system, msn.maya.rig.components.ik.LegIKSystem):
                        ik_twist_attributes = leg_twist_attributes
                    else:
                        continue  # Unrecognized ik type?!

                    for attribute in ik_twist_attributes:
                        pose_attributes = self._ik_controllers.get(ik_system.control).get("attrs")
                        if attribute in pose_attributes:
                            value = pose_attributes.get(attribute).get("value")
                            shared.maya.attribute.set_attribute(ik_system.control, attribute, value)

                            if keyframe:
                                maya.cmds.setKeyframe(ik_system.control, attribute=attribute, value=value)



    def keyCommonGimbalNodes(self, rig):
        """
        The switch from DG to Parallel evaluation mode and while keying nodes on an animation layer,
        has an annoying glitch of re-evaluating the transforms of certain nodes when Parallel mode is re-enabled.

        It seems to be something with keying individual transform channels one at a time as hitting the whole transform
        group with a setKeyframe() operation allows the nodes to hold their position, after parallel mode is re-enabled
        """
        if not isinstance(rig, msn.maya.rig.types.character.Rig):
            return

        gimbal_nodes = ["{}:{}".format(rig.namespace, b) for b in
                        [rig.bone_map.clavicle_left, rig.bone_map.clavicle_right, rig.bone_map.hip_left, rig.bone_map.hip_right]]

        posing_nodes = []

        for pair in self._cache:
            if pair[0] and pair[1]:
                posing_nodes.append(pair[1].name())

        for node in gimbal_nodes:
            if node in posing_nodes:
                for attribute in shared.maya.attribute.list_keyable_transforms(node):
                    maya.cmds.setKeyframe(node, attribute=attribute)

    def isCharacterRig(self, rig):
        return isinstance(rig, msn.maya.rig.types.character.Rig)

    def isPosingRig(self):
        return bool(self._rig_list)

    def isIkSystemPosing(self, ik_system):
        """
        Test if we are posing either a part of a rig's IK system, or a parent of that ik system.  This is intended to test
        if an IK system needs to be switched to FK during posing.

        Args:
            ik_system (msn.maya.rig.components.ik.IKSystem): Ik system to test with.

        Returns (bool): True if either an FK bone of an ik system, or a parent of that system, is being posed.
                        False if neither the ik system or any of its parent nodes are part of the active posing nodes.
        """

        for bone in ik_system.bone_list:
            bone_parents = shared.maya.hierarchy.list_hierarchy(bone)
            posing_nodes = set([i[1].name() for i in self._cache])
            if any([b in posing_nodes for b in bone_parents + [bone]]):
                return True
        return False

    def applyRelativeTo(self, relativeTo):
        cog, cog_data = self.getDataNodeByName('cog')
        root, root_data = self.getDataNodeByName('root')

        if not cog or not root:
            logger.warning("Pose is missing data for bones needed to apply relatively to the Root, falling back to non-relative behavior.")
            print("Bones missing from pose: {}".format(",".join([i.split(':')[-1] for i in [cog, root] if i])))
            return

        self._relative_to_snapshot[cog] = copy.deepcopy(cog_data)
        self._relative_to_snapshot[root] = copy.deepcopy(root_data)

        cog_pose_world_matrix = om2.MMatrix(cog_data["attrs"]["worldMatrix"]["value"])
        root_pose_world_matrix = om2.MMatrix(root_data["attrs"]["worldMatrix"]["value"])

        root_current_local_matrix = shared.maya.api.matrix.get_matrix(root, "matrix")
        root_current_world_matrix = shared.maya.api.matrix.get_matrix(root, "matrix")
        root_current_parent_matrix = shared.maya.api.matrix.get_matrix(root, "parentMatrix")

        cog_current_local_matrix = shared.maya.api.matrix.get_matrix(cog, "matrix")
        cog_current_world_matrix = shared.maya.api.matrix.get_matrix(cog, "worldMatrix")
        cog_current_parent_matrix = shared.maya.api.matrix.get_matrix(cog, "parentMatrix")

        if relativeTo.lower() == 'root':
            world_matrix = (cog_pose_world_matrix * root_pose_world_matrix.inverse()) * root_current_local_matrix
            local_matrix = world_matrix * cog_current_parent_matrix.inverse()

            cog_data["attrs"]["matrix"]["value"] = shared.maya.api.matrix.matrix_as_list(local_matrix)
            cog_data["attrs"]["worldMatrix"]["value"] = shared.maya.api.matrix.matrix_as_list(world_matrix)

            root_data["attrs"]["matrix"]["value"] = shared.maya.api.matrix.matrix_as_list(root_current_local_matrix)
            root_data["attrs"]["worldMatrix"]["value"] = shared.maya.api.matrix.matrix_as_list(root_current_world_matrix)

        elif relativeTo.lower() == 'cog':
            world_matrix = (root_pose_world_matrix * cog_pose_world_matrix.inverse()) * cog_current_world_matrix
            local_matrix = world_matrix * root_current_parent_matrix.inverse()

            root_data["attrs"]["worldMatrix"]["value"] = shared.maya.api.matrix.matrix_as_list(world_matrix)
            root_data["attrs"]["matrix"]["value"] = shared.maya.api.matrix.matrix_as_list(local_matrix)

            cog_data["attrs"]["worldMatrix"]["value"] = shared.maya.api.matrix.matrix_as_list(cog_current_world_matrix)
            cog_data["attrs"]["matrix"]["value"] = shared.maya.api.matrix.matrix_as_list(cog_current_local_matrix)

    def gunRelativeTo(self, relativeTo):
        gun, gun_data = self.getDataNodeByName('gun_ctrl')
        cog, cog_data = self.getDataNodeByName('cog')

        if not gun or not cog:
            logger.warning("Pose is missing data for bones needed to apply relatively to the Root, falling back to non-relative behavior.")
            print("Bones missing from pose: {}".format(",".join([i.split(':')[-1] for i in [gun, cog] if i])))
            return

        self._gun_relative_to_snapshot[gun] = copy.deepcopy(gun_data)

        gun_pose_world_matrix = om2.MMatrix(gun_data["attrs"]["worldMatrix"]["value"])
        cog_pose_world_matrix = om2.MMatrix(cog_data["attrs"]["worldMatrix"]["value"])

        gun_current_parent_matrix = shared.maya.api.matrix.get_matrix(gun, "parentMatrix")
        cog_current_world_matrix = shared.maya.api.matrix.get_matrix(cog, "worldMatrix")

        if relativeTo.lower() == 'character':
            world_matrix = (gun_pose_world_matrix * cog_pose_world_matrix.inverse()) * cog_current_world_matrix
            local_matrix = world_matrix * gun_current_parent_matrix.inverse()

            gun_data["attrs"]["worldMatrix"]["value"] = shared.maya.api.matrix.matrix_as_list(world_matrix)
            gun_data["attrs"]["matrix"]["value"] = shared.maya.api.matrix.matrix_as_list(local_matrix)

    def getDataNodeByName(self, name):
        """ Fetch a node by name from the posing data.  This is a namespace-less search."""
        for node in self._data.get("objects").keys():
            if name.lower() == node.split(":")[-1].lower():
                return node, self._data.get("objects").get(node)
        return None, None

    def beforeLoad(self, clearSelection=True):
        """
        Called before loading the pose.
        
        :type clearSelection: bool
        """
        logger.debug('Before Load "%s"', self.path())

        if not self._isLoading:
            self._isLoading = True
            maya.cmds.undoInfo(openChunk=True)

            self._selection = maya.cmds.ls(selection=True) or []
            self._autoKeyFrame = maya.cmds.autoKeyframe(query=True, state=True)

            maya.cmds.autoKeyframe(edit=True, state=False)
            maya.cmds.select(clear=clearSelection)

    def afterLoad(self):
        """Called after loading the pose."""
        if not self._isLoading:
            return

        logger.debug("After Load '%s'", self.path())

        self._isLoading = False
        if self._selection:
            maya.cmds.select(self._selection)
            self._selection = None

        maya.cmds.autoKeyframe(edit=True, state=self._autoKeyFrame)
        maya.cmds.undoInfo(closeChunk=True)

        logger.debug('Loaded "%s"', self.path())

    def refreshSceneState(self):
        maya.cmds.refresh(currentView=True)

        '''
        TODO: Test if this is more stable in Maya 2022.  The visible cache
              can get stuck, and flushing the cache can resolve the issue.
              However, Maya is very unstable when flushing the geometry cache,
              so we might have to live with a glitchy cache unless this action
              is more stable in newer versions of Maya.
        '''
        #shared.maya.animation.cache.invalidate_playback_range()

    def hasTransforms(self, attrs_list):
        for attr in attrs_list:
            if self.isTransform(attr):
                return True
        return False

    def isTransform(self, attr):
        return attr in ["translateX", "translateY", "translateZ",
                        "rotateX", "rotateY", "rotateZ",
                        "scaleX", "scaleY", "scaleZ"]

    def isRotationTransform(self, attr):
        return attr in ["rotateX", "rotateY", "rotateZ"]

    def hasMatrixTransforms(self):
        """
        Test if the current pose contains matrix data for posing transforms.  Without matrix transforms, the internal _value attribute
        will be used to apply the pose.  Otherwise, the matrix will be used to pose any transform attributes.  If ANY transform attribute
        does not have matrix data saved with it, this function will return False.  There shouldn't be any new poses without matrix information
        but this will ensure that legacy behavior (eg using _value and not matrices) is used, over failing to apply the pose entirely.

        Returns (bool): True if local space matrices are found along side the stored attributes.
                        False if matrices are not found (legacy poses)
        """

        for idx, data in enumerate(self.cache()):
            srcAttribute, dstAttribute, srcMirrorValue = data

            if self.isTransform(srcAttribute.attr()):
                matrix, worldMatrix = self.matrixFromCache(self.cache(), srcAttribute.name())
                if not matrix:
                    return False
        return True

    def updateValuesFromMatrices(self):
        """
        Before applying a pose using matrices, the pose is silently applied in its entirety,
        the attribute channels are captured, and the original locations are then restored.
        """

        matrix_update = {}
        for idx, data in enumerate(self.cache()):
            srcAttribute, dstAttribute, srcMirrorValue = data

            if self.isTransform(srcAttribute.attr()):
                current_matrix = maya.cmds.xform(srcAttribute.name(), matrix=True, query=True)
                matrix_update[srcAttribute.name()] = current_matrix

        for idx, data in enumerate(self.cache()):
            srcAttribute, dstAttribute, srcMirrorValue = data

            if self.isTransform(srcAttribute.attr()):
                matrix, worldMatrix = self.matrixFromCache(self.cache(), srcAttribute.name())
                if matrix:
                    maya.cmds.xform(srcAttribute.name(), matrix=matrix, worldSpace=False)

        for idx, data in enumerate(self.cache()):
            srcAttribute, dstAttribute, srcMirrorValue = data

            if self.isTransform(srcAttribute.attr()):
                value = maya.cmds.getAttr(srcAttribute.fullname())
                # if self.isRotationTransform(srcAttribute.attr()):
                #      value = shared.python.math.unwind_360(value)
                srcAttribute.setValue(value)

        for node, matrix in matrix_update.items():
            maya.cmds.xform(node, matrix=matrix, worldSpace=False)

    def matrixFromCache(self, cache, node):
        matrix, worldMatrix = None, None
        for srcAttribute, dstAttribute, srcMirrorValue in cache:
            if srcAttribute.name() != node:
                continue
            if srcAttribute.attr() == "matrix":
                matrix = srcAttribute.value()
            elif srcAttribute.attr() == "worldMatrix":
                worldMatrix = srcAttribute.value()
        return matrix, worldMatrix

    @mutils.timing
    @shared.maya.decorators.undo
    @shared.maya.decorators.disable_auto_keyframe
    @shared.maya.decorators.restore_selection
    @shared.maya.decorators.as_dg
    def load(
            self,
            objects=None,
            namespaces=None,
            attrs=None,
            blend=100,
            key=False,
            mirror=False,
            additive=False,
            refresh=False,
            batchMode=False,
            clearCache=False,
            mirrorTable=None,
            onlyConnected=False,
            clearSelection=False,
            ignoreConnected=False,
            searchAndReplace=None,
            applyRelativeTo=None,
            gunRelativeTo=None
    ):
        """
        Load the pose to the given objects or namespaces.
        
        :type objects: list[str]
        :type namespaces: list[str]
        :type attrs: list[str]
        :type blend: float
        :type key: bool
        :type refresh: bool
        :type mirror: bool
        :type additive: bool
        :type mirrorTable: mutils.MirrorTable
        :type batchMode: bool
        :type clearCache: bool
        :type ignoreConnected: bool
        :type onlyConnected: bool
        :type clearSelection: bool
        :type searchAndReplace: (str, str) or None
        :type applyRelativeTo: str or None
        """
        if mirror and not mirrorTable:
            logger.warning("Cannot mirror pose without a mirror table!")
            mirror = False

        if batchMode:
            key = False

        self._namespaces = namespaces  # Update the TransferObject's namespace list

        self.updateRigList()

        if self.isPosingRig():
            self.appendFkSystems(objects)

        self.updateCache(
            objects=objects,
            namespaces=namespaces,
            attrs=attrs,
            batchMode=batchMode,
            clearCache=clearCache,
            mirrorTable=mirrorTable,
            onlyConnected=onlyConnected,
            ignoreConnected=ignoreConnected,
            searchAndReplace=searchAndReplace,
            applyRelativeTo=applyRelativeTo,
            gunRelativeTo=gunRelativeTo
        )

        if self.isPosingRig():
            self.captureRigStates()
            self.setRigsToPosing(keyframe=key)

        if self.version() > "1.0.0":
            self.updateValuesFromMatrices()

        maya.cmds.select(clear=True)

        try:
            self.loadCache(blend=blend, key=key, mirror=mirror,
                           additive=additive)
        finally:
            if not batchMode:

                if self.isPosingRig():

                    if key:
                        for rig in self._rig_list:
                            self.keyCommonGimbalNodes(rig)

                    if applyRelativeTo:
                        for key, value in self._relative_to_snapshot.items():
                            self._data.get("objects")[key] = value
                        self._relative_to_snapshot = dict()

                    if gunRelativeTo:
                        for key, value in self._gun_relative_to_snapshot.items():
                            self._data.get("objects")[key] = value
                        self._gun_relative_to_snapshot = dict()

                    self.restoreRigStates()
                    self.loadIkTwistValues(keyframe=key)
                    self.resetRigList()

                # Return the focus to the Maya window
                maya.cmds.setFocus("MayaWindow")

        if refresh:
            self.refreshSceneState()

    def updateCache(
            self,
            objects=None,
            namespaces=None,
            attrs=None,
            ignoreConnected=False,
            onlyConnected=False,
            mirrorTable=None,
            batchMode=False,
            clearCache=True,
            searchAndReplace=None,
            applyRelativeTo=None,
            gunRelativeTo=None
    ):
        """
        Update the pose cache.
        
        :type objects: list[str] or None 
        :type namespaces: list[str] or None
        :type attrs: list[str] or None
        :type ignoreConnected: bool
        :type onlyConnected: bool
        :type clearCache: bool
        :type batchMode: bool
        :type mirrorTable: mutils.MirrorTable
        :type searchAndReplace: (str, str) or None
        :type applyRelativeTo: str or None
        """
        if clearCache or not batchMode or not self._mtime:
            self._mtime = self.mtime()

        if self.isPosingRig():
            if applyRelativeTo or gunRelativeTo:
                if self.version() > "1.0.0":
                    if applyRelativeTo:
                        self.applyRelativeTo(applyRelativeTo)
                    if gunRelativeTo and not self._gun_relative_to_snapshot:
                        self.gunRelativeTo(gunRelativeTo)
                else:
                    logging.warning("This pose must be re-saved to allow for relative posing, falling back to non-relative behavior.")

            ik_controllers = []
            pole_vector_controllers = []

            for rig in self._rig_list:
                for ik_system in rig.ik_systems:
                    ik_controllers.append(shared.maya.namespace.strip_namespace(ik_system.control))
                    pole_vector_controllers.append(shared.maya.namespace.strip_namespace(ik_system.pole_vector_control))

            for node in self._data.get("objects").keys():
                node_no_ns = shared.maya.namespace.strip_namespace(node)

                if node_no_ns in ik_controllers + pole_vector_controllers:
                    if node_no_ns in ik_controllers:
                        self._ik_controllers[node] = self._data["objects"].pop(node)
                    else:
                        self._data["objects"].pop(node)

        mtime = self._mtime

        cacheKey = \
            str(mtime) + \
            str(objects) + \
            str(attrs) + \
            str(namespaces) + \
            str(ignoreConnected) + \
            str(searchAndReplace) + \
            str(maya.cmds.currentTime(query=True))

        if self._cacheKey != cacheKey or clearCache:

            self.validate(namespaces=namespaces)

            self._cache = []
            self._cacheKey = cacheKey

            dstObjects = objects
            srcObjects = self.objects()
            usingNamespaces = not objects and namespaces

            if mirrorTable:
                self.setMirrorTable(mirrorTable)

            search = None
            replace = None
            if searchAndReplace:
                search = searchAndReplace[0]
                replace = searchAndReplace[1]

            matches = mutils.matchNames(
                srcObjects,
                dstObjects=dstObjects,
                dstNamespaces=namespaces,
                search=search,
                replace=replace,
            )

            for srcNode, dstNode in matches:
                self.cacheNode(
                    srcNode,
                    dstNode,
                    attrs=attrs,
                    onlyConnected=onlyConnected,
                    ignoreConnected=ignoreConnected,
                    usingNamespaces=usingNamespaces,
                )

        if not self.cache():
            text = "No objects match when loading data. " \
                   "Turn on debug mode to see more details."

            logger.error(mutils.NoMatchFoundError(text))

    def cacheNode(
            self,
            srcNode,
            dstNode,
            attrs=None,
            ignoreConnected=None,
            onlyConnected=None,
            usingNamespaces=None
    ):
        """
        Cache the given pair of nodes.
        
        :type srcNode: mutils.Node
        :type dstNode: mutils.Node
        :type attrs: list[str] or None 
        :type ignoreConnected: bool or None
        :type onlyConnected: bool or None
        :type usingNamespaces: none or list[str]
        """
        mirrorAxis = None
        mirrorObject = None

        # Remove the first pipe in-case the object has a parent
        dstNode.stripFirstPipe()

        srcName = srcNode.name()

        if self.mirrorTable():
            mirrorObject = self.mirrorTable().mirrorObject(srcName)

            if not mirrorObject:
                mirrorObject = srcName
                msg = "Cannot find mirror object in pose for %s"
                logger.debug(msg, srcName)

            # Check if a mirror axis exists for the mirrorObject otherwise
            # check the srcNode
            mirrorAxis = self.mirrorAxis(mirrorObject) or self.mirrorAxis(srcName)

            if mirrorObject and not maya.cmds.objExists(mirrorObject):
                msg = "Mirror object does not exist in the scene %s"
                logger.debug(msg, mirrorObject)

        if usingNamespaces:
            # Try and use the short name.
            # Much faster than the long name when setting attributes.
            try:
                dstNode = dstNode.toShortName()
            except mutils.NoObjectFoundError as msg:
                logger.debug(msg)
                return
            except mutils.MoreThanOneObjectFoundError as msg:
                logger.debug(msg)
                return

        for attr in self.attrs(srcName):

            if attrs and attr not in attrs:
                continue

            dstAttribute = mutils.Attribute(dstNode.name(), attr)
            isConnected = dstAttribute.isConnected()

            if (ignoreConnected and isConnected) or (onlyConnected and not isConnected):
                continue

            type_ = self.attrType(srcName, attr)
            value = self.attrValue(srcName, attr)
            srcMirrorValue = self.mirrorValue(mirrorObject, attr, mirrorAxis=mirrorAxis)

            srcAttribute = mutils.Attribute(dstNode.name(), attr, value=value, type=type_)
            dstAttribute.update()

            self._cache.append((srcAttribute, dstAttribute, srcMirrorValue))

    def loadCache(self, blend=100, key=False, mirror=False, additive=False):
        """
        Load the pose from the current cache.
        
        :type blend: float
        :type key: bool
        :type mirror: bool
        :rtype: None
        """
        cache = self.cache()

        for idx, data in enumerate(cache):
            srcAttribute, dstAttribute, srcMirrorValue = data

            if srcAttribute.attr() in ["matrix", "worldMatrix"]:
                continue

            if srcAttribute and dstAttribute:
                if mirror and srcMirrorValue is not None:
                    value = srcMirrorValue
                else:
                    value = srcAttribute.value()

                try:
                    dstAttribute.set(value, blend=blend, key=key,
                                     additive=additive)
                except (ValueError, RuntimeError):
                    cache[idx] = (None, None)
                    logger.debug('Ignoring %s', dstAttribute.fullname())
