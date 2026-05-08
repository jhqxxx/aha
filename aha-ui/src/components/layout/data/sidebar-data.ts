import {
  Package,
  Play,
  BookOpen,
  Settings,
  UserCog,
  Wrench,
  Palette,
  Bell,
  Monitor,

} from 'lucide-react'
import { type SidebarData } from '../types'

export const sidebarData: SidebarData = {
  user: {
    name: 'AHA User',
    email: 'user@aha.app',
    avatar: '/avatars/shadcn.jpg',
  },
  teams: [
    {
      name: 'AHA Launcher',
      logo: Play,
      plan: '模型推理服务启动器',
    },
  ],
  navGroups: [
    {
      title: '导航',
      items: [
        {
          title: '模型列表',
          url: '/',
          icon: Package,
        },
        {
          title: '启动服务',
          url: '/launch',
          icon: Play,
        },
        {
          title: '使用指南',
          url: '/usage',
          icon: BookOpen,
        },
      ],
    },
    {
      title: '设置',
      items: [
        {
          title: 'Settings',
          icon: Settings,
          items: [
            {
              title: 'Model',
              url: '/settings/model',
              icon: Package,
            },
            {
              title: 'Profile',
              url: '/settings',
              icon: UserCog,
            },
            {
              title: 'Account',
              url: '/settings/account',
              icon: Wrench,
            },
            {
              title: 'Appearance',
              url: '/settings/appearance',
              icon: Palette,
            },
            {
              title: 'Notifications',
              url: '/settings/notifications',
              icon: Bell,
            },
            {
              title: 'Display',
              url: '/settings/display',
              icon: Monitor,
            },
          ],
        },
      ],
    },
  ],
}
