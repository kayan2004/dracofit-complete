import {
  Injectable,
  BadRequestException,
  NotFoundException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, In } from 'typeorm';
import { Friendship, FriendshipStatus } from './entities/friendship.entity'; // Removed FriendAction from here
import { CreateFriendshipDto } from './dto/create-friendship.dto';
import { UpdateFriendshipDto } from './dto/update-friendship.dto';
import { User } from '../users/entities/user.entity';
import { FriendshipFilter } from './dto/friendship-filter.dto';
import {
  FriendAction,
  FriendActionType,
} from '../friend_actions/friend_action.entity'; // Import both the entity and the enum type

@Injectable()
export class FriendshipsService {
  constructor(
    @InjectRepository(Friendship)
    private friendshipRepository: Repository<Friendship>,
    @InjectRepository(User)
    private userRepository: Repository<User>,
    @InjectRepository(FriendAction)
    private friendActionRepository: Repository<FriendAction>, // Correct repository
  ) {}

  async create(
    createFriendshipDto: CreateFriendshipDto,
    currentUserId: number,
  ) {
    // Prevent self-friending
    if (currentUserId === createFriendshipDto.friendId) {
      throw new BadRequestException(
        'You cannot send a friend request to yourself',
      );
    }

    // Check if users exist
    const [user1, user2] = await Promise.all([
      this.userRepository.findOneOrFail({ where: { id: currentUserId } }),
      this.userRepository.findOneOrFail({
        where: { id: createFriendshipDto.friendId },
      }),
    ]);

    // Check if friendship already exists
    const existingFriendship = await this.friendshipRepository.findOne({
      where: [
        {
          user1: { id: currentUserId },
          user2: { id: createFriendshipDto.friendId },
        },
        {
          user1: { id: createFriendshipDto.friendId },
          user2: { id: currentUserId },
        },
      ],
    });

    if (existingFriendship) {
      throw new BadRequestException('Friendship already exists');
    }

    // Create new friendship
    const newFriendship = new Friendship();
    newFriendship.user1 = user1;
    newFriendship.user2 = user2;
    newFriendship.status = FriendshipStatus.PENDING;

    return this.friendshipRepository.save(newFriendship);
  }

  async findAll(userId: number) {
    return this.friendshipRepository.find({
      where: [{ user1: { id: userId } }, { user2: { id: userId } }],
      relations: ['user1', 'user2'],
    });
  }

  async findOne(id: number, userId: number) {
    const friendship = await this.friendshipRepository.findOne({
      where: [
        { id, user1: { id: userId } },
        { id, user2: { id: userId } },
      ],
      relations: ['user1', 'user2'],
    });

    if (!friendship) {
      throw new NotFoundException('Friendship not found');
    }

    return friendship;
  }

  async update(
    id: number,
    updateFriendshipDto: UpdateFriendshipDto,
    userId: number,
  ) {
    const friendship = await this.findOne(id, userId);

    // Only the recipient (user2) can accept the friendship
    if (friendship.user2.id !== userId) {
      throw new BadRequestException(
        'Only the friend request recipient can accept/reject',
      );
    }

    friendship.status = updateFriendshipDto.status;
    return this.friendshipRepository.save(friendship);
  }

  async remove(id: number, userId: number) {
    const friendship = await this.findOne(id, userId);
    return this.friendshipRepository.remove(friendship);
  }

  async getPendingRequests(userId: number) {
    return this.friendshipRepository.find({
      where: {
        user2: { id: userId },
        status: FriendshipStatus.PENDING,
      },
      relations: ['user1'],
    });
  }

  async getFriends(userId: number) {
    const friendships = await this.friendshipRepository.find({
      where: [
        { user1: { id: userId }, status: FriendshipStatus.ACCEPTED },
        { user2: { id: userId }, status: FriendshipStatus.ACCEPTED },
      ],
      relations: ['user1', 'user2'],
    });

    return friendships.map((friendship) =>
      friendship.user1.id === userId ? friendship.user2 : friendship.user1,
    );
  }

  async getFriendshipsByStatus(userId: number, filter: FriendshipFilter) {
    switch (filter) {
      case FriendshipFilter.PENDING:
        return this.friendshipRepository.find({
          where: {
            user2: { id: userId },
            status: FriendshipStatus.PENDING,
          },
          relations: ['user1'],
        });

      case FriendshipFilter.ACCEPTED:
        const friendships = await this.friendshipRepository.find({
          where: [
            { user1: { id: userId }, status: FriendshipStatus.ACCEPTED },
            { user2: { id: userId }, status: FriendshipStatus.ACCEPTED },
          ],
          relations: ['user1', 'user2'],
        });
        return friendships.map((friendship) =>
          friendship.user1.id === userId ? friendship.user2 : friendship.user1,
        );

      case FriendshipFilter.ALL:
      default:
        return this.friendshipRepository.find({
          where: [{ user1: { id: userId } }, { user2: { id: userId } }],
          relations: ['user1', 'user2'],
        });
    }
  }

  async logFriendAction(userId: number, action: FriendActionType) {
    const user = await this.userRepository.findOne({ where: { id: userId } });
    if (!user) {
      throw new NotFoundException('User not found');
    }

    const friendAction = this.friendActionRepository.create({
      user,
      action, // This 'action' now matches the expected enum type
    });

    return this.friendActionRepository.save(friendAction);
  }

  async getFriendActions(userId: number) {
    return this.friendActionRepository.find({
      where: { user: { id: userId } },
      order: { createdAt: 'DESC' },
    });
  }
}
